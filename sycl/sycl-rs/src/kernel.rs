//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::{Result, usmbox::MemoryUses};
use bytemuck::Pod;
use sycl_rs_sys::{kernel_bundle::ffi, types};

/// A kernel bundle which stores loaded SYCL source code.
pub struct SourceKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::SourceKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::SourceKernelBundle>> for SourceKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::SourceKernelBundle>) -> Self {
        Self(value)
    }
}

impl SourceKernelBundle {
    pub fn build(&mut self) -> Result<ExecutableKernelBundle> {
        ffi::build(&mut self.0).map(Into::into)
    }
}

/// A kernel bundle which stores compiled SYCL kernels.
pub struct ExecutableKernelBundle(pub(crate) cxx::UniquePtr<types::ffi::ExecutableKernelBundle>);

impl From<cxx::UniquePtr<types::ffi::ExecutableKernelBundle>> for ExecutableKernelBundle {
    fn from(value: cxx::UniquePtr<types::ffi::ExecutableKernelBundle>) -> Self {
        Self(value)
    }
}

impl ExecutableKernelBundle {
    pub fn get_kernel(&mut self, name: &str) -> Result<Kernel> {
        ffi::get_kernel(&mut self.0, name).map(Into::into)
    }
}

/// An executable SYCL kernel.
pub struct Kernel(pub(crate) cxx::UniquePtr<types::ffi::Kernel>);

impl From<cxx::UniquePtr<types::ffi::Kernel>> for Kernel {
    fn from(value: cxx::UniquePtr<types::ffi::Kernel>) -> Self {
        Self(value)
    }
}

/// Types which can be passed as SYCL kernel arguments.
///
/// # Safety
///
/// A type implementing this trait must mirror the representation and alignment of the
/// corresponding SYCL kernel argument structure. Its [`as_raw_arg`](KernelArgument::as_raw_arg)
/// must pass `uses` on to the `as_raw_arg` of every [`UsmBox`](crate::usmbox::UsmBox) it refers
/// to, and any other memory it refers to must stay allocated until the kernel completes.
pub unsafe trait KernelArgument {
    /// Converts self to a raw byte representation, and records the USM memory it refers to, if
    /// any, in `uses`.
    ///
    /// # Safety
    ///
    /// This function returns a reference to raw bytes. These bytes will be passed to FFI
    /// functions. The caller must make sure these functions respect Rust's aliasing rules.
    unsafe fn as_raw_arg<'a>(&'a self, uses: &mut MemoryUses<'a>) -> &'a [u8];
}

unsafe impl<T: Pod> KernelArgument for T {
    unsafe fn as_raw_arg<'a>(&'a self, _uses: &mut MemoryUses<'a>) -> &'a [u8] {
        bytemuck::bytes_of(self)
    }
}

/// Types which describe an argument list for a SYCL kernel.
///
/// # Safety
///
/// A type implementing this trait must mirror the representation and alignment of each
/// corresponding SYCL kernel argument inside the returned array, and its
/// [`as_raw_arg_list`](KernelArgumentList::as_raw_arg_list) must pass `uses` on to the
/// [`KernelArgument::as_raw_arg`] of each argument.
pub unsafe trait KernelArgumentList<const ARGC: usize> {
    /// Converts each struct member to a raw byte representation, and records the USM memory
    /// they refer to in `uses`.
    ///
    /// # Safety
    ///
    /// This function returns references to raw bytes. These bytes will be passed to FFI
    /// functions. The caller must make sure these functions respect Rust's aliasing rules.
    unsafe fn as_raw_arg_list<'a>(&'a self, uses: &mut MemoryUses<'a>) -> [&'a [u8]; ARGC];
}

unsafe impl KernelArgumentList<0> for () {
    unsafe fn as_raw_arg_list<'a>(&'a self, _uses: &mut MemoryUses<'a>) -> [&'a [u8]; 0] {
        []
    }
}

unsafe impl<T: KernelArgument> KernelArgumentList<1> for T {
    unsafe fn as_raw_arg_list<'a>(&'a self, uses: &mut MemoryUses<'a>) -> [&'a [u8]; 1] {
        [unsafe { self.as_raw_arg(uses) }]
    }
}

pub use sycl_rs_derive::KernelArgumentList;

use sycl_rs_derive::impl_arg_list_for_tuples;

impl_arg_list_for_tuples! {16}
