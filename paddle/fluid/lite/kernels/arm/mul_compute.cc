// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/lite/kernels/arm/mul_compute.h"
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

/// (1,2,3,4) , x_num_col_dims = 2  -> (2,12)
// flatten_to_2d
DDim flatten_to_2d(const DDim &src, int num_col_dims) {
  int rank = src.size();
  return DDim{{src.Slice(0, num_col_dims).production(),
               src.Slice(num_col_dims, rank).production()}};
}

inline Tensor ReshapeToMatrix(const Tensor &src, int num_col_dims) {
  Tensor res;
  res.ShareDataWith(src);
  res.Resize(flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

void MulCompute::Run() {
  auto &param = this->Param<operators::MulParam>();
  const lite::Tensor *x = param.x;
  const lite::Tensor *y = param.y;
  lite::Tensor *output = param.output;

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  auto output_dims = output->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(y_dims.size(), 2UL);
  CHECK_EQ(output_dims.size(), 2UL);

  auto x_num_col_dims = param.x_num_col_dims;
  auto y_num_col_dims = param.y_num_col_dims;

  const lite::Tensor x_matrix =
      x->dims().size() > 2 ? ReshapeToMatrix(*x, x_num_col_dims) : *x;
  const lite::Tensor y_matrix =
      y->dims().size() > 2 ? ReshapeToMatrix(*y, y_num_col_dims) : *y;
  if (output_dims.size() != 2) {
    output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

#if 0
  const auto* x_data = param.x->data<float>();
  const auto* y_data = param.y->data<float>();
#endif

  const auto *x_data = x_matrix.data<float>();
  const auto *y_data = y_matrix.data<float>();
  auto *o_data = param.output->mutable_data<float>();

#if 0  // check x_matrix, y_matrix
  std::cout << "x_matrix ---- " << std::endl;
  auto x_matrix_dims = x_matrix.dims();
  for (int i=0; i< x_matrix_dims.size(); ++i) {
    std::cout << x_matrix_dims[i] << std::endl;
  }

  std::cout << "y_matrix ---- " << std::endl;
  auto y_matrix_dims = y_matrix.dims();
  for (int i=0; i< y_matrix_dims.size(); ++i) {
    std::cout << y_matrix_dims[i] << std::endl;
  }
#endif

#if 0
  int x_h = x_dims.Slice(0, param.x_num_col_dims).production();
  int x_w = x_dims.Slice(param.x_num_col_dims, x_dims.size()).production();
  int y_h = y_dims.Slice(0, param.y_num_col_dims).production();
  int y_w = y_dims.Slice(param.y_num_col_dims, y_dims.size()).production();
  LOG(INFO) << "x_h:" << x_h;
  LOG(INFO) << "x_w:" << x_w;
  LOG(INFO) << "y_h:" << y_h;
  LOG(INFO) << "y_w:" << y_w;
#endif

#if 0
  // If refer paddle-mobile:
  output->mutable_data<float>();
  math::MatMul<float, float>(x_matrix, false, y_matrix, false,
                             static_cast<float>(1), output,
                             static_cast<float>(0));

  if (output_dims.size() != 2) {
    output->Resize(output_dims);
  }
#else
// if refer anakin: ./saber/lite/funcs/neon/saber_matmul.cpp dispatch
// ignored.
#endif
}

TargetType MulCompute::target() const { return TARGET(kARM); }

PrecisionType MulCompute::precision() const { return PRECISION(kFloat); }

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul, kARM, kFloat, kNCHW,
                     paddle::lite::kernels::arm::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
