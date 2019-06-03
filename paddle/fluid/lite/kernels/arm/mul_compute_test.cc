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
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define A(i, p) a[i * lda + p]
#define B(p, j) b[p * ldb + j]
#define C(i, j) c[i * ldc + j]
void gemm(int m, int n, int k, int lda, int ldb, int ldc, float* a, float* b,
          float* c) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      C(i, j) = 0;
      for (int p = 0; p < k; ++p) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

void print_matrix(std::string mat_name, float* mat, int rows, int cols) {
  std::cout << "mat_name:" << mat_name << std::endl;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << mat[r * cols + c] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n\n" << std::endl;
}

TEST(mul_arm, retrive_op) {
  auto mul =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("mul");
  ASSERT_FALSE(mul.empty());
  ASSERT_TRUE(mul.front());
}

TEST(mul_arm, init) {
  MulCompute mul;
  ASSERT_EQ(mul.precision(), PRECISION(kFloat));
  ASSERT_EQ(mul.target(), TARGET(kARM));
}

TEST(mul_arm, compare_test) {
  // x: <m, k>
  // y: <k, n>
  // output: <m, n>
  // ref: <m, n>
  const int m = 2;
  const int n = 3;
  const int k = 4;
  const int lda = k;
  const int ldb = n;
  const int ldc = n;

  LOG(INFO) << "m:" << m;
  LOG(INFO) << "n:" << n;
  LOG(INFO) << "k:" << k;
  LOG(INFO) << "lda:" << lda;
  LOG(INFO) << "ldb:" << ldb;
  LOG(INFO) << "ldc:" << ldc;

  lite::Tensor x, y, output, ref;
  constexpr int batch_size = n;
  x.Resize({m, batch_size});
  y.Resize({batch_size, k});
  output.Resize({n, k});
  ref.Resize({n, k});

  auto* x_data = x.mutable_data<float>();
  auto* y_data = y.mutable_data<float>();
  auto* output_data = output.mutable_data<float>();
  auto* ref_data = ref.mutable_data<float>();

  LOG(INFO) << "x.dims().product():" << x.dims().product();
  LOG(INFO) << "y.dims().product():" << y.dims().product();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < y.dims().product(); i++) {
    y_data[i] = static_cast<float>(i);
  }

  print_matrix("a(x)", x_data, m, k);
  print_matrix("b(y)", y_data, k, n);
  print_matrix("c(output)", output_data, m, n);
  print_matrix("c1(ref)", ref_data, m, n);

  gemm(m, n, k, lda, ldb, ldc, x_data, y_data, ref_data);

  print_matrix("c(output)", output_data, m, n);
  print_matrix("c1(ref)", ref_data, m, n);

#if 0
  // TODO(YS): mul test
  b_data = nullptr;
  lite::arm::math::mul_compute_eigen(x_data, batch_size, 3,  //
                                     w_data, 3, 4,           //
                                     b_data, ref_data);
#endif

  // mul compute kernel
  MulCompute mul;
  operators::MulParam param;

  param.x = &x;
  param.y = &y;
  param.output = &output;
  param.x_num_col_dims = 1;
  param.y_num_col_dims = 1;

  DeviceInfo::init_info();
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  mul.SetParam(param);
  mul.SetContext(std::move(ctx));
  mul.Run();
}

#if 0
TEST(mul_arm, compare_test) {
  lite::Tensor x, w, b, out, ref;
  constexpr int batch_size = 2;
  x.Resize({batch_size, 3});
  w.Resize({3, 4});
  b.Resize({1, 4});
  out.Resize({batch_size, 4});
  ref.Resize({batch_size, 4});

  auto x_data = x.mutable_data<float>();
  auto w_data = w.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto ref_data = ref.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().product(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().product(); i++) {
    b_data[i] = static_cast<float>(i);
  }

  // TODO(TJ): enable bias soon
  b_data = nullptr;
  lite::arm::math::mul_compute_eigen(x_data, batch_size, 3,  //
                                    w_data, 3, 4,           //
                                    b_data, ref_data);

  // mul compute kernel
  MulCompute mul;
  operators::MulParam param;

  param.in_num_col_dims = 1;
  param.input = &x;
  param.w = &w;
  param.bias = nullptr;
  param.output = &out;
  param.in_mat_dims = x.dims();

  DeviceInfo::init_info();
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  mul.SetParam(param);
  mul.SetContext(std::move(ctx));
  mul.Run();

  VLOG(3) << "output vs ref";
  for (int i = 0; i < out.dims().product(); i++) {
    VLOG(3) << out_data[i] << " vs " << ref_data[i];
  }

  for (int i = 0; i < out.dims().product(); ++i) {
    EXPECT_NEAR(out_data[i], ref_data[i], 1e-5);
  }
}

TEST(mul_arm, num_col_dims) {
  MulCompute mul;
  operators::MulParam param;

  lite::Tensor x;
  lite::Tensor w;
  lite::Tensor bias;
  lite::Tensor output;

  x.Resize({1, 2, 3});
  w.Resize({3, 4});
  bias.Resize({1, 4});
  output.Resize({2, 4});

  auto* x_data = x.mutable_data<float>();
  auto* w_data = w.mutable_data<float>();
  auto* bias_data = bias.mutable_data<float>();
  auto* output_data = output.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().product(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().product(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < bias.dims().product(); i++) {
    bias_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < output.dims().product(); i++) {
    output_data[i] = static_cast<float>(i);
  }

  param.in_num_col_dims = 2;
  param.input = &x;
  param.w = &w;
  param.bias = &bias;
  param.output = &output;
  param.in_mat_dims = x.dims();

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  DeviceInfo::init_info();

  mul.SetParam(param);
  mul.SetContext(std::move(ctx));
  mul.Run();
}

#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
