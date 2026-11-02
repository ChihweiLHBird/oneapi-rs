//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use sycl_rs::prelude::*;

static SLOW_FILL_SRC: &str = r#"
#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void slow_fill(unsigned int count, unsigned int *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    for (unsigned int i = 1; i <= count; ++i)
        *(volatile unsigned int *)(ptr + id) = i;
}
"#;

// Reading a UsmBox from the host waits for the kernel that writes it, even when the launch event
// is dropped without waiting.
#[test]
fn host_access_waits_for_kernel() -> sycl_rs::Result<()> {
    const LEN: usize = 16;
    // Enough work that the kernel is still running when the host reads the box.
    const COUNT: u32 = 50_000_000;
    let mut queue = Queue::new();
    let mut array = queue.alloc_shared::<u32>(LEN)?.wait()?;

    let kernel = queue
        .get_context()
        .create_kernel_bundle_from_source(SLOW_FILL_SRC)?
        .build()?
        .get_kernel("slow_fill")?;
    let event = unsafe {
        queue.launch(
            NdRange::new([LEN as u64], [LEN as u64]),
            &kernel,
            (COUNT, &mut array),
        )
    }?;
    drop(event);

    assert_eq!(*array, [COUNT; LEN]);
    Ok(())
}
