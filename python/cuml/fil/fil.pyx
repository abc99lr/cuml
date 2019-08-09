#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import copy
import cudf
import ctypes
import math
import numpy as np
import warnings

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteLoadXGBoostModel(const char* filename,
                                      ModelHandle* out)
    cdef int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf,
                                                      size_t len,
                                                      ModelHandle* out)
    cdef int TreeliteFreeModel(ModelHandle handle)
    cdef int TreeliteQueryNumTree(ModelHandle handle, size_t* out)
    cdef int TreeliteQueryNumFeature(ModelHandle handle, size_t* out)
    cdef int TreeliteLoadLightGBMModel(const char* filename, ModelHandle* out)
    cdef int TreeliteLoadProtobufModel(const char* filename, ModelHandle* out)


cdef class TreeliteModel():
    """
    Wrapper for Treelite-loaded forest

    This is only used for loading saved models into FIL, it is
    not used directly for inference.

    Attributes
    ----------

    handle : ModelHandle
        Opaque pointer to Treelite model
    """
    cpdef ModelHandle handle

    def __cinit__(self):
        self.handle = <ModelHandle>NULL

    cdef set_handle(self, ModelHandle new_handle):
        self.handle = new_handle

    cdef ModelHandle get_handle(self):
        return self.handle

    def __cdel__(self):
        if self.handle != NULL:
            TreeliteFreeModel(self.handle)

    @property
    def num_trees(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumTree(self.handle, &out)
        return out

    @property
    def num_features(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumFeature(self.handle, &out)
        return out

    @staticmethod
    def from_filename(filename, model_type):
        """
        Returns a TreeliteModel object loaded from `filename`

        Parameters
        ----------
        filename : string
            Path to treelite model file to load

        model_type : string
            Type of model: 'xgboost', 'protobuf', or 'lightgbm'
        """
        filename_bytes = filename.encode("UTF-8")
        cdef ModelHandle handle
        if model_type == "xgboost":
            res = TreeliteLoadXGBoostModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        elif model_type == "protobuf":
            # XXX Not tested
            res = TreeliteLoadProtobufModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        elif model_type == "lightgbm":
            # XXX Not tested
            res = TreeliteLoadLightGBMModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        else:
            raise ValueError("Unknown model type %s" % model_type)
        model = TreeliteModel()
        model.set_handle(handle)
        return model


cdef extern from "fil/fil.h" namespace "ML::fil":
    cdef enum algo_t:
        NAIVE,
        TREE_REORG,
        BATCH_TREE_REORG

    cdef enum output_t:
        RAW,
        PROB,
        CLASS

    cdef struct dense_node_t:
        float val
        int bits

    cdef struct forest:
        pass

    ctypedef forest* forest_t

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold

    cdef void free(cumlHandle& handle,
                   forest_t)

    cdef void predict(cumlHandle& handle,
                      forest_t,
                      float*,
                      float*,
                      size_t)

    cdef forest_t from_treelite(cumlHandle& handle,
                                forest_t*,
                                ModelHandle,
                                treelite_params_t*)

cdef class FIL_impl():

    cpdef object handle
    cdef forest_t forest_data

    def __cinit__(self,
                  handle=None):
        self.handle = handle

    def predict(self, X, preds=None):
        """
        Returns the results of forest inference on the exampes in X

        Parameters
        ----------
        X : float32 array-like (device or host) shape = (n_samples, n_features)
            For optimal performance, pass a device array with C-style layout

        preds : float32 device array, shape = n_samples
        """
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, _, X_dtype = \
            input_to_dev_array(X, order='C', check_dtype=np.float32)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if preds is None:
            preds = cuda.device_array(n_rows, dtype=np.float32)
        elif (not isinstance(preds, cudf.Series) and
              not cuda.is_cuda_array(preds)):
                raise ValueError("Invalid type for output preds,"
                                 " need GPU array")

        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = input_to_dev_array(
            preds,
            check_dtype=np.float32)

        predict(handle_[0],
                self.forest_data,
                <float*> preds_ptr,
                <float*> X_ptr,
                <size_t> n_rows)
        self.handle.sync()
        # synchronous w/o a stream
        return preds

    def load_from_treelite_model(self,
                                 TreeliteModel model,
                                 bool output_class,
                                 algo_t algo,
                                 float threshold):
        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = threshold
        treelite_params.algo = algo
        self.forest_data = NULL
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t>model.handle
        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        return self

    def __cdel__(self):
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        free(handle_[0],
             self.forest_data)
        return self


class FIL(Base):
    """
    Parameters
    ----------
    handle : cuml.Handle
       If it is None, a new one is created just for this class.
    """
    def __init__(self,
                 handle=None):
        super(FIL, self).__init__(handle)
        self._impl = FIL_impl(self.handle)

    def predict(self, X, preds=None):
        """
        Predicts the labels for X with the loaded forest model

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
           For optimal performance, pass a device array with C-style layout
        preds: gpuarray or cudf.Series, shape = (n_samples,)
           Optional 'out' location to store inference results

        Returns
        ----------
        GPU array of length n_samples with inference results
        (or 'preds' filled with inference results if preds was specified)
        """
        return self._impl.predict(X, preds)

    def load_from_treelite_model(self, model, output_class,
                                 algo=algo_t.TREE_REORG,
                                 threshold=0.5):
        """
        Creates a FIL model using the treelite model
        passed to the function.

        Parameters
        ----------
        model : the model information in the treelite format
           loaded from a saved model using the treelite API
           https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class: boolean
           If true, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        algo : int (from algo_t enum)
             0 (NAIVE) - simple inference using shared memory
             1 (TREE_REORG) - similar to naive but trees rearranged to be more
                              coalescing-friendly
             2 (BATCH_TREE_REORG) - similar to TREE_REORG but predicting
                                    multiple rows per thread block
        threshold : threshold is used to for classification
           applied if output == OUTPUT_CLASS, else it is ignored
        """
        return self._impl.load_from_treelite_model(model, output_class,
                                                   algo, threshold)

    @staticmethod
    def from_treelite_file(filename,
                           output_class=False,
                           threshold=0.50,
                           algo=algo_t.TREE_REORG,
                           model_type="xgboost",
                           handle=None):
        """
        Returns a FIL instance containing the forest saved in 'filename'
        This uses Treelite to load the saved model.

        Parameters
        ----------
        filename : str
           Path to saved model file in a treelite-compatible format
           (See https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class : bool
           If true, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        threshold : float
           Cutoff value above which a prediction is set to 1.0
           Only used if the model is classificaiton and output_class is True
        algo : algo_t
           Which inference algorithm to use.
           See documentation in FIL.load_from_treelite_model
        model_type : str
            Format of saved treelite model to load.
            Can be 'xgboost', 'lightgbm', or 'protobuf'
        """
        cuml_fm = FIL(handle=handle)
        tl_model = TreeliteModel.from_filename(filename, model_type=model_type)
        cuml_fm.load_from_treelite_model(tl_model,
                                         algo=algo,
                                         output_class=output_class,
                                         threshold=threshold)
        return cuml_fm