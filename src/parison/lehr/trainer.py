from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from parison.annealing.objectives import kiln_loss
from parison.bench.logbook import scribe
from parison.bench.seed import kindle
from parison.bench.settings import Recipe
from parison.furnace.twin import TherapyTwin
from parison.gather.streams import Parcel, collate
from parison.gather.windows import Foundry
from parison.lehr.ledger import stamp
from parison.lehr.schedule import cosine_with_warmup

_LOG = scribe("parison.lehr")


class Glassworks:
    def __init__(self, recipe: Recipe, device: torch.device, workdir: Path) -> None:
        self.recipe = recipe
        self.device = device
        self.workdir = workdir
        self.gen = kindle(recipe.bench.seed)
        self.train_set = Foundry(recipe.feed, recipe.glass, recipe.bench.seed, "train")
        self.val_set = Foundry(recipe.feed, recipe.glass, recipe.bench.seed, "val")
        self.val_set.attach_norm(self.train_set.norm)
        self.model = TherapyTwin(recipe.glass, recipe.feed).to(device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=recipe.oven.lr,
            betas=recipe.oven.betas,
            weight_decay=recipe.oven.weight_decay,
        )
        steps = max(1, len(self._loader(self.train_set, True)))
        total = max(1, recipe.oven.max_epochs * steps // max(1, recipe.bench.grad_accum))
        self.sched = cosine_with_warmup(self.opt, recipe.oven.warmup_steps, total)
        self.use_amp = device.type == "cuda" and recipe.bench.precision == "fp16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _loader(self, dataset: Foundry, shuffle: bool) -> DataLoader[Parcel]:
        return DataLoader(
            dataset,
            batch_size=self.recipe.bench.per_device_batch,
            shuffle=shuffle,
            collate_fn=collate,
            drop_last=False,
            generator=self.gen,
        )

    def _forward(self, batch: Parcel) -> torch.Tensor:
        outputs, _ = self.model(batch)
        total, report = kiln_loss(outputs, batch, self.recipe.oven)
        self._last = report
        return total

    def run_step(self, batch: Parcel) -> dict[str, float]:
        self.model.train()
        batch = batch.to(self.device)
        self.opt.zero_grad()
        loss = self._forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.oven.grad_clip)
        self.opt.step()
        self.sched.step()
        return self._last

    def _epoch(self) -> float:
        self.model.train()
        loader = self._loader(self.train_set, True)
        accum = self.recipe.bench.grad_accum
        self.opt.zero_grad()
        running = 0.0
        seen = 0
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self._forward(batch) / accum
            self.scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.oven.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                self.sched.step()
            running += self._last["total"]
            seen += 1
        return running / max(1, seen)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        loader = self._loader(self.val_set, False)
        running = 0.0
        seen = 0
        for batch in loader:
            batch = batch.to(self.device)
            self._forward(batch)
            running += self._last["total"]
            seen += 1
        return running / max(1, seen)

    def fit(self, max_epochs: int | None = None) -> float:
        budget = max_epochs if max_epochs is not None else self.recipe.oven.max_epochs
        best = float("inf")
        stale = 0
        for epoch in range(budget):
            train_loss = self._epoch()
            val_loss = self.validate()
            _LOG.info("epoch %d train=%.4f val=%.4f", epoch, train_loss, val_loss)
            if val_loss < best - 1e-4:
                best = val_loss
                stale = 0
                stamp(
                    self.workdir / "best.ckpt",
                    self.model,
                    self.opt,
                    epoch,
                    self.recipe.bench.seed,
                    {"norm": self.train_set.norm.stats, "val": val_loss},
                )
            else:
                stale += 1
                if stale >= self.recipe.oven.early_stop_patience:
                    _LOG.info("early stop at epoch %d", epoch)
                    break
        return best
