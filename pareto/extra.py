# https://github.com/tinygrad/tinygrad/blob/master/extra/lr_scheduler.py
from tinygrad.nn.optim import Optimizer
from tinygrad.tensor import Tensor

class LR_Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.epoch_counter = Tensor([0], device=self.optimizer.device)

    def get_lr(self):
        pass

    def schedule_step(self) -> list[Tensor]:
        return [self.epoch_counter.assign(self.epoch_counter + 1), self.optimizer.lr.assign(self.get_lr())]

    def step(self) -> None:
        Tensor.realize(*self.schedule_step())

class OneCycleLR(LR_Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        div_factor: float,
        final_div_factor: float,
        total_steps: int,
        pct_start: float,
        anneal_strategy: str = "linear",
        cycle_momentum: bool = False,
    ):
        super().__init__(optimizer)
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor
        self.total_steps = total_steps
        self.pct_start = pct_start
        assert anneal_strategy == "linear", "only linear annealing supported"
        assert not cycle_momentum, "cycle momentum not supported"
        self.optimizer.lr.assign(self.get_lr()).realize()  # update the initial LR

    @staticmethod
    def _annealing_linear(start: float, end: float, pct: Tensor) -> Tensor:
        return pct * (end - start) + start

    def get_lr(self) -> Tensor:
        return (
            (self.epoch_counter < self.total_steps * self.pct_start)
            .where(
                self._annealing_linear(self.initial_lr, self.max_lr, self.epoch_counter / (self.total_steps * self.pct_start)),
                self._annealing_linear(
                    self.max_lr, self.min_lr, (self.epoch_counter - (self.total_steps * self.pct_start)) / (self.total_steps * (1 - self.pct_start))
                ),
            )
            .cast(self.optimizer.lr.dtype)
        )
