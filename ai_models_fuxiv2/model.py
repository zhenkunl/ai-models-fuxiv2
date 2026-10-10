# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import os

import numpy as np
import torch
from ai_models.model import Model
import pandas as pd
import xarray as xr
import inspect

CANONICAL_ORDER = ["input", "step", "hour", "doy"]
LOG = logging.getLogger(__name__)


class FuXiv2(Model):
    download_url = "https://huggingface.co/tpys/fuxi-2.1/resolve/main/{file}"
    download_files = ["fuxi-2.1.pt2", "mean.nc", "std.nc"]

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    param_sfc = ["msl", "2t", "2d", "sst", "10u", "10v", "100u", "100v", "lcc", "mcc", "hcc", "tcc", "ssr", "ssrd", "fdir", "ttr", "tcw", "tp"]

    param_level_pl = (
        ["z", "t", "u", "v", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    # Output
    expver = "fxv2"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.hour_steps = 6
        self.lagged = [-6, 0]

        self.ordering = [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ] + self.param_sfc

    def get_init_time(self):
        init_time = self.all_fields.order_by(valid_datetime="descending")[0].datetime()["base_time"]
        init_time = pd.to_datetime(init_time)
        return init_time

    def load_norm_stats(self):
        mean_file = os.path.join(self.assets, self.download_files[1])
        self.mean = xr.open_dataarray(mean_file).values[:, None, None]
        std_file = os.path.join(self.assets, self.download_files[2])
        self.std = xr.open_dataarray(std_file).values[:, None, None]

    def normalize(self, input, mean, std, exp_channel=-1):
        """normalize input data"""
        if exp_channel is not None:
            ch = input[..., exp_channel, :, :]
            np.clip(ch, 0, None, out=ch)
            np.log1p(ch, out=ch)

        np.subtract(input, mean, out=input)
        np.divide(input, std + 1e-8, out=input)

        return input

    def postprocess(self, output, mean, std, exp_channel=-1):
        """Denormalize output in-place: output = output * std + mean.

        The tp channel (last, index 84) was log1p-normalized during training,
        so expm1 is applied to reverse it. Pass exp_channel=None to skip.

        Args:
            output: array with shape (..., C, H, W)
            mean: shape (C,)
            std: shape (C,)
            exp_channel: channel index for expm1 reversal (-1 = last)
        """
        np.multiply(output, std, out=output)
        np.add(output, mean, out=output)

        if exp_channel is not None:
            ch = output[..., exp_channel, :, :]
            np.clip(ch, None, 20, out=ch)
            np.expm1(ch, out=ch)
            np.clip(ch, 0, None, out=ch)

        return output

    def get_input(self):
        fields_pl = self.fields_pl
        param, level = self.param_level_pl
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by('valid_datetime', param=param, level=level)
        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy1, fields_pl_numpy2 = np.split(fields_pl_numpy, 2, axis=0)
        # scaled ×1000 from ERA5 kg/kg to g/kg
        fields_pl_numpy1[52:] = fields_pl_numpy1[52:] * 1000
        fields_pl_numpy2[52:] = fields_pl_numpy2[52:] * 1000

        fields_sfc = self.fields_sfc
        fields_sfc = fields_sfc.sel(param=self.param_sfc)
        fields_sfc = fields_sfc.order_by('valid_datetime', param=self.param_sfc)
        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)
        fields_sfc_numpy1, fields_sfc_numpy2 = np.split(fields_sfc_numpy, 2, axis=0)
        # calculate wind speed from u and v components
        ws10m = np.hypot(fields_sfc_numpy1[4], fields_sfc_numpy1[5])
        ws100m = np.hypot(fields_sfc_numpy1[6], fields_sfc_numpy1[7])
        fields_sfc_numpy1 = np.concatenate([fields_sfc_numpy1[:4], ws10m[None], ws100m[None], fields_sfc_numpy1[4:]], axis=0)
        ws10m = np.hypot(fields_sfc_numpy2[4], fields_sfc_numpy2[5])
        ws100m = np.hypot(fields_sfc_numpy2[6], fields_sfc_numpy2[7])
        fields_sfc_numpy2 = np.concatenate([fields_sfc_numpy2[:4], ws10m[None], ws100m[None], fields_sfc_numpy2[4:]], axis=0)

        prev = np.concatenate([fields_pl_numpy1, fields_sfc_numpy1], axis=0)
        curr = np.concatenate([fields_pl_numpy2, fields_sfc_numpy2], axis=0)
        input = np.stack([prev, curr], axis=0)

        self.template = fields_pl[len(fields_pl) // len(self.lagged) :] + fields_sfc[len(fields_sfc) // len(self.lagged) :]

        return input

    def load_model(self):
        model_path = os.path.join(self.assets, self.download_files[0])
        # with torch.cuda.device(self.device):
        ep = torch.export.load(model_path)
        module = ep.module()
        del ep

        sig = inspect.signature(module.forward)
        sig_names = list(sig.parameters.keys())
        if sig_names and sig_names[0].startswith("args_"):
            input_names = CANONICAL_ORDER[: len(sig_names)]
        else:
            input_names = sig_names

        # Infer dtype from model parameters
        dtype = torch.float32
        for p in module.parameters():
            dtype = p.dtype
            break

        torch.cuda.empty_cache()
        return module, input_names, dtype

    def prepare_features(self, valid_time, step, dtype):
        tod = (valid_time.hour * 60 + valid_time.minute) / 1440.0
        doy = min(365, valid_time.day_of_year) / 365.0
        return {
            "step": torch.tensor([step], dtype=dtype, device=self.device),
            "hour": torch.tensor([tod], dtype=dtype, device=self.device),
            "doy": torch.tensor([doy], dtype=dtype, device=self.device),
        }

    def run(self):

        self.load_norm_stats()
        module, input_names, dtype = self.load_model()
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        input = self.get_input()
        input = self.normalize(input, self.mean, self.std)
        state = torch.from_numpy(input[None].astype(np_dtype)).to(self.device)
        forecast_time = self.get_init_time()
        dt = pd.Timedelta(hours=self.hour_steps)

        with torch.no_grad():
            with self.stepper(6) as stepper:
                for i in range(self.lead_time // self.hour_steps):
                    step = (i + 1) * self.hour_steps

                    features = self.prepare_features(forecast_time + i * dt, i, dtype)

                    # Build positional args
                    ordered_args = [state]
                    for name in CANONICAL_ORDER[1:]:
                        if name in input_names and name in features:
                            ordered_args.append(features[name])

                    # Run model
                    result = module(*ordered_args)

                    # Extract prediction: last frame of the 2-frame output
                    prediction = result[:, -1].clone().cpu().float().numpy()  # (1, 85, 721, 1440) float32

                    # Denormalize
                    self.postprocess(prediction, self.mean, self.std)

                    # Convert back to kg/kg
                    prediction[:, 52:65] = prediction[:, 52:65] / 1000.0

                    # Remove wind speed at 10m and 100m (indices 69 and 70) from the output, as they can be derived from u and v components
                    prediction = np.concatenate([prediction[:, :69], prediction[:, 71:]], axis=1)

                    state = result

                    for k, fs in enumerate(self.template):
                        self.write(prediction[0, k, ...], template=fs, step=step)

                    stepper(i, step)