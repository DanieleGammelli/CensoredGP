# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import GP
from .. import likelihoods
from .. import kern
import numpy as np
from ..inference.latent_function_inference.expectation_propagation import EP, EPCensored

class GPCensoredRegression(GP):
    """
    Gaussian Process Censored Regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values, can be None if likelihood is not None
    :param censoring: indicator function explicitating whether the observation is censored or not
    :param kernel: a GPy kernel, defaults to rbf
    :param likelihood: a GPy likelihood, defaults to CensoredGaussian
    :param inference_method: Latent function inference to use, defaults to EP
    :type inference_method: :class:`GPy.inference.latent_function_inference.LatentFunctionInference`


    """

    def __init__(self, X, Y, censoring, kernel=None, Y_metadata=None, mean_function=None, inference_method=None,
                 likelihood=None, normalizer=False):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if likelihood is None:
            likelihood = likelihoods.CensoredGaussian(censoring=censoring)

        if inference_method is None:
            inference_method = EPCensored()

        GP.__init__(self, X=X, Y=Y, kernel=kernel, likelihood=likelihood, inference_method=inference_method,
                    mean_function=mean_function, name='gp_censored_regression', normalizer=normalizer)

        self.censoring = censoring
        inference_method = EPCensored()
        print("defaulting to " + str(inference_method) + " for latent function inference")

        self.inference_method = inference_method

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        GPCensoredRegression(gp.X, gp.Y, gp.cens, gp.kern, gp.likelihood, gp.inference_method,
                             gp.mean_function, name='gp_censored_regression')

    def to_dict(self, save_data=True):
        model_dict = super(GPCensoredRegression,self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPCensoredRegression"
        return model_dict

    @staticmethod
    def from_dict(input_dict, data=None):
        import GPy
        m = GPy.core.model.Model.from_dict(input_dict, data)
        return GPCensoredRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        input_dict = GPCensoredRegression._format_input_dict(input_dict, data)
        input_dict.pop('name', None)  # Name parameter not required by GPClassification
        return GPCensoredRegression(**input_dict)

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.censoring, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)
