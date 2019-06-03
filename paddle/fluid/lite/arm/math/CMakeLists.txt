
cc_library(math_arm SRCS funcs.cc packed_sgemm.cc gemv.cc DEPS ${lite_kernel_deps} eigen3)
