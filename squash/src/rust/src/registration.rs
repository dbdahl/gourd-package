// Generated by the cargo::prebuild() function. Do not edit by hand.

// If usage of .Call() and .Kall() functions in the package's R code changes,
// update this file by rerunning "cargo::prebuild(DIR)", where DIR is the root
// directory of this package.

/*
// Below is skeleton code that you can copy to your "src/rust/src/lib.rs" file
// and then uncomment. You can freely change the body and arguments names, but
// changing the name of a function or the number of arguments necessitates:
// 1. a corresponding change to invocations of .Call() and .Kall() in the R code
// and 2. rerunning cargo::prebuild().

mod registration;
use roxido::*;

#[roxido]
fn data_r2rust(data: Rval, missingItems: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn state_r2rust(state: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn hyperparameters_r2rust(hyperparameters: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn fit(burnin: Rval, data: Rval, state: Rval, fixed: Rval, hyperparameters: Rval, partitionDistribution: Rval, missingItems: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn log_likelihood_contributions(state: Rval, data: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn log_likelihood_contributions_of_missing(state: Rval, data: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn state_rust2r_as_reference(state: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn rust_free(data: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn all(all: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn fit_all(all_ptr: Rval, shrinkage: Rval, nIterations: Rval, doBaselinePartition: Rval) -> Rval {
    Rval::nil()
}

#[roxido]
fn sample_multivariate_normal(n: Rval, mean: Rval, precision: Rval) -> Rval {
    Rval::nil()
}
*/

use roxido::*;

#[no_mangle]
extern "C" fn R_init_squash_rust(info: *mut rbindings::DllInfo) {
    let mut call_routines = Vec::with_capacity(11);
    let mut _names: Vec<std::ffi::CString> = Vec::with_capacity(11);
    _names.push(std::ffi::CString::new(".data_r2rust").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::data_r2rust as *const u8) },
        numArgs: 2,
    });
    _names.push(std::ffi::CString::new(".state_r2rust").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::state_r2rust as *const u8) },
        numArgs: 1,
    });
    _names.push(std::ffi::CString::new(".hyperparameters_r2rust").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::hyperparameters_r2rust as *const u8) },
        numArgs: 1,
    });
    _names.push(std::ffi::CString::new(".fit").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::fit as *const u8) },
        numArgs: 7,
    });
    _names.push(std::ffi::CString::new(".log_likelihood_contributions").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::log_likelihood_contributions as *const u8) },
        numArgs: 2,
    });
    _names.push(std::ffi::CString::new(".log_likelihood_contributions_of_missing").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::log_likelihood_contributions_of_missing as *const u8) },
        numArgs: 2,
    });
    _names.push(std::ffi::CString::new(".state_rust2r_as_reference").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::state_rust2r_as_reference as *const u8) },
        numArgs: 1,
    });
    _names.push(std::ffi::CString::new(".rust_free").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::rust_free as *const u8) },
        numArgs: 1,
    });
    _names.push(std::ffi::CString::new(".all").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::all as *const u8) },
        numArgs: 1,
    });
    _names.push(std::ffi::CString::new(".fit_all").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::fit_all as *const u8) },
        numArgs: 4,
    });
    _names.push(std::ffi::CString::new(".sample_multivariate_normal").unwrap());
    call_routines.push(rbindings::R_CallMethodDef {
        name: _names.last().unwrap().as_ptr(),
        fun: unsafe { std::mem::transmute(crate::sample_multivariate_normal as *const u8) },
        numArgs: 3,
    });
    call_routines.push(rbindings::R_CallMethodDef {
        name: std::ptr::null(),
        fun: None,
        numArgs: 0,
    });
    unsafe {
        rbindings::R_registerRoutines(
            info,
            std::ptr::null(),
            call_routines.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
        );
        rbindings::R_useDynamicSymbols(info, 0);
        rbindings::R_forceSymbols(info, 1);
    }
    roxido::r::set_custom_panic_hook();
}
