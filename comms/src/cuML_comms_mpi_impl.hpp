/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <mpi.h>

#include <common/cuml_comms_iface.hpp>

namespace ML {
    
class cumlMPICommunicator_impl : public MLCommon::cumlCommunicator_iface {
public:
    cumlMPICommunicator_impl() =delete;

    cumlMPICommunicator_impl(MPI_Comm comm);

    virtual ~cumlMPICommunicator_impl();

    virtual int getSize() const;
    virtual int getRank() const;

    virtual void barrier() const;

    virtual void isend(const void *buf, std::size_t size, int dest, int tag, request_t *request) const;

    virtual void irecv(void *buf, std::size_t size, int source, int tag, request_t *request) const;

    virtual void waitall(int count, request_t array_of_requests[]) const;

private:
    MPI_Comm                                            _mpi_comm;
#ifdef HAVE_NCCL
    ncclComm_t                                          _nccl_comm;
#endif
    int                                                 _size;
    int                                                 _rank;
    mutable request_t                                   _next_request_id;
    mutable std::unordered_map<request_t,MPI_Request>   _requests_in_flight;
    mutable std::unordered_set<request_t>               _free_requests;
};

} // end namespace ML