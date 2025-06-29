import typing
from abc import ABC, abstractmethod
from typing import final
import torch
from torch import nn
from pypose.optim.functional import modjac
from pypose.optim.strategy import TrustRegion
from pypose.optim.solver import Cholesky
from pypose.optim.corrector import FastTriggs
from pypose.optim.optimizer import _Optimizer, Trivial, RobustModel


class FactorGraph(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def write_back(self) -> typing.Any: ...

    @abstractmethod
    def covariance_array(self) -> torch.Tensor: ...


class AnalyticModule(nn.Module, ABC):
    verify: bool = False
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_forward_pre_hook(self.record_forward_args)
        self._call_args = None

    def record_forward_args(self, module, input):
        """
        Record the arguments passed to the forward() method.
        """
        self._call_args = input

    @torch.no_grad()
    @abstractmethod
    def build_jacobian(self) -> torch.Tensor:
        """
        This function should be implemented by the user.
        It should return the jacobian of the model's previous forward call with respect to model's parameters.
        Should be called only after forward() has been called.
        """
        pass 
    
    @final
    @torch.no_grad()  
    def jacobian(self) -> torch.Tensor:
        """
        Returns the jacobian of the model's previous forward call with respect to model's parameters.
        Should be called only after forward() has been called.
        """
        J_analytic = self.build_jacobian()
        if self.verify:
            assert self.verify_jacobian(J_analytic), "Analytic Jacobian from build_jacobian() does not match with autograd jacobian!"
        return J_analytic
        

    @torch.no_grad()
    def verify_jacobian(self, J_analytic: torch.Tensor) -> bool: 
        """
        Verifies whether the input J_analytic coincides with autograd jacobian of the previous forward call. 
        Should be called only after forward() has been called.
        """
        assert self._call_args is not None, "Jacobian verification failed! No forward call was made."
        J_autograd = modjac(self, input=self._call_args, flatten=False, vectorize=True)
        params = dict(self.named_parameters())
        params_values = tuple(params.values())
        if isinstance(J_autograd, (tuple, list)):
            J_autograd = torch.cat([j.reshape(-1, p.numel()) for j, p in zip(J_autograd, params_values)], 1)
        J_autograd = torch.cat(J_autograd) if isinstance(J_autograd, (tuple, list)) else J_autograd
        return torch.allclose(J_analytic, J_autograd)


class LM_autograd(_Optimizer):
    def __init__(self, model: nn.Module, solver=None, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)
        self.jackwargs = {'vectorize': vectorize}
        self.solver = Cholesky() if solver is None else solver
        self.reject, self.reject_count = reject, 0
        self.weight = weight
        if kernel is not None:
            kernel = [kernel] if not isinstance(kernel, (tuple, list)) else kernel
            kernel = [k if k is not None else Trivial() for k in kernel]
            self.corrector = [FastTriggs(k) for k in kernel] if corrector is None else corrector
        else:
            self.corrector = [Trivial()] if corrector is None else corrector
        self.corrector = [self.corrector] if not isinstance(self.corrector, (tuple, list)) else self.corrector
        self.corrector = [c if c is not None else Trivial() for c in self.corrector]
        self.model = RobustModel(model, kernel)


    @torch.no_grad()
    def step(self, input, target=None, weight=None):    #type: ignore
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))
            J = modjac(self.model, input=(input, target), flatten=False, **self.jackwargs)  # type: ignore
            params = dict(self.model.named_parameters())
            params_values = tuple(params.values())
            J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            for i in range(len(R)):
                R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
                    else self.corrector[i](R = R[i], J = J[i])
            R, weight, J = self.model.normalize_RWJ(R, weight, J)

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            J_T = J.T @ weight if weight is not None else J.T
            A, self.reject_count = J_T @ J, 0
            A.diagonal().clamp_(pg['min'], pg['max'])
            while self.last <= self.loss:
                A.diagonal().add_(A.diagonal() * pg['damping'])
                try:
                    D = self.solver(A = A, b = -J_T @ R.view(-1, 1))
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss


class LM_analytic(_Optimizer):
    def __init__(self, model: AnalyticModule, solver=None, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)
        self.jackwargs = {'vectorize': vectorize}
        self.solver = Cholesky() if solver is None else solver
        self.reject, self.reject_count = reject, 0
        self.weight = weight
        if kernel is not None:
            kernel = [kernel] if not isinstance(kernel, (tuple, list)) else kernel
            kernel = [k if k is not None else Trivial() for k in kernel]
            self.corrector = [FastTriggs(k) for k in kernel] if corrector is None else corrector
        else:
            self.corrector = [Trivial()] if corrector is None else corrector
        self.corrector = [self.corrector] if not isinstance(self.corrector, (tuple, list)) else self.corrector
        self.corrector = [c if c is not None else Trivial() for c in self.corrector]
        self.model = RobustModel(model, kernel)


    @torch.no_grad()
    def step(self, input, target=None, weight=None):    #type: ignore
        for pg in self.param_groups:
            weight_mtr = self.weight if weight is None else weight
            R = self.model(input, target=target)
            J = self.model.model.jacobian()
            R = torch.cat(R, dim=0)
            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0
            R, J = self.corrector[0](R, J)
            J_T = J.mT 
            if weight is None:
                A = J_T @ J
                b = -J_T @ R.view(-1, 1)
            else:
                J_T = J_T @ weight_mtr
                A = J_T  @ J
                b = - J_T  @ R.view(-1, 1)
            A.diagonal().clamp_(pg['min'], pg['max'])
            while self.last <= self.loss:
                A.diagonal().add_(A.diagonal() * pg['damping'])
                try:
                    D = self.solver(A = A, b = -J_T @ R.view(-1, 1))
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
            
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
