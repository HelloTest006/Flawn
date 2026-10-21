#include "flawn_pybridge.hpp"
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace flawn::py {

// Minimal CPython 3 Stable ABI types & function pointers (zero header conflict with MinGW)
using PyObject = void;
using Py_ssize_t = intptr_t;

struct CPythonAPI {
    void (*Py_InitializeEx)(int) = nullptr;
    int (*Py_IsInitialized)() = nullptr;
    int (*Py_FinalizeEx)() = nullptr;
    PyObject* (*PyImport_ImportModule)(const char*) = nullptr;
    PyObject* (*PyImport_AddModule)(const char*) = nullptr;
    PyObject* (*PyModule_GetDict)(PyObject*) = nullptr;
    PyObject* (*PyObject_GetAttrString)(PyObject*, const char*) = nullptr;
    PyObject* (*PyObject_CallObject)(PyObject*, PyObject*) = nullptr;
    PyObject* (*PyTuple_New)(Py_ssize_t) = nullptr;
    int (*PyTuple_SetItem)(PyObject*, Py_ssize_t, PyObject*) = nullptr;
    PyObject* (*PyLong_FromLongLong)(long long) = nullptr;
    long long (*PyLong_AsLongLong)(PyObject*) = nullptr;
    PyObject* (*PyFloat_FromDouble)(double) = nullptr;
    double (*PyFloat_AsDouble)(PyObject*) = nullptr;
    PyObject* (*PyUnicode_FromString)(const char*) = nullptr;
    const char* (*PyUnicode_AsUTF8)(PyObject*) = nullptr;
    PyObject* (*PyObject_Str)(PyObject*) = nullptr;
    int (*PyRun_SimpleString)(const char*) = nullptr;
    PyObject* (*PyRun_String)(const char*, int, PyObject*, PyObject*) = nullptr;
    int (*PyDict_SetItemString)(PyObject*, const char*, PyObject*) = nullptr;
    void (*PyErr_Print)() = nullptr;
    PyObject* (*PyErr_Occurred)() = nullptr;
    void (*Py_DecRef)(PyObject*) = nullptr;
};

static CPythonAPI g_py;
constexpr int PY_EVAL_INPUT = 258;
constexpr int PY_FILE_INPUT = 257;

PythonBridge::PythonBridge() = default;

PythonBridge::~PythonBridge() {
    // Keep interpreter alive till process exit to avoid C-extension teardown issues
}

bool PythonBridge::initialize() {
    if (initialized_) return true;

#if defined(_WIN32)
    // Try standard PATH first, then user's Python 3.14 installation path
    HMODULE hMod = LoadLibraryA("python314.dll");
    if (!hMod) {
        const char* py_dir = "C:\\Users\\user\\AppData\\Local\\Python\\pythoncore-3.14-64";
        SetDllDirectoryA(py_dir);
        std::string full_dll = std::string(py_dir) + "\\python314.dll";
        hMod = LoadLibraryA(full_dll.c_str());
    }
    if (!hMod) return false;
    lib_handle_ = reinterpret_cast<void*>(hMod);

    auto load = [&](const char* sym) -> void* {
        return reinterpret_cast<void*>(GetProcAddress(hMod, sym));
    };

    g_py.Py_InitializeEx       = reinterpret_cast<decltype(g_py.Py_InitializeEx)>(load("Py_InitializeEx"));
    g_py.Py_IsInitialized      = reinterpret_cast<decltype(g_py.Py_IsInitialized)>(load("Py_IsInitialized"));
    g_py.Py_FinalizeEx         = reinterpret_cast<decltype(g_py.Py_FinalizeEx)>(load("Py_FinalizeEx"));
    g_py.PyImport_ImportModule = reinterpret_cast<decltype(g_py.PyImport_ImportModule)>(load("PyImport_ImportModule"));
    g_py.PyImport_AddModule    = reinterpret_cast<decltype(g_py.PyImport_AddModule)>(load("PyImport_AddModule"));
    g_py.PyModule_GetDict      = reinterpret_cast<decltype(g_py.PyModule_GetDict)>(load("PyModule_GetDict"));
    g_py.PyObject_GetAttrString= reinterpret_cast<decltype(g_py.PyObject_GetAttrString)>(load("PyObject_GetAttrString"));
    g_py.PyObject_CallObject   = reinterpret_cast<decltype(g_py.PyObject_CallObject)>(load("PyObject_CallObject"));
    g_py.PyTuple_New           = reinterpret_cast<decltype(g_py.PyTuple_New)>(load("PyTuple_New"));
    g_py.PyTuple_SetItem       = reinterpret_cast<decltype(g_py.PyTuple_SetItem)>(load("PyTuple_SetItem"));
    g_py.PyLong_FromLongLong   = reinterpret_cast<decltype(g_py.PyLong_FromLongLong)>(load("PyLong_FromLongLong"));
    g_py.PyLong_AsLongLong     = reinterpret_cast<decltype(g_py.PyLong_AsLongLong)>(load("PyLong_AsLongLong"));
    g_py.PyFloat_FromDouble    = reinterpret_cast<decltype(g_py.PyFloat_FromDouble)>(load("PyFloat_FromDouble"));
    g_py.PyFloat_AsDouble      = reinterpret_cast<decltype(g_py.PyFloat_AsDouble)>(load("PyFloat_AsDouble"));
    g_py.PyUnicode_FromString  = reinterpret_cast<decltype(g_py.PyUnicode_FromString)>(load("PyUnicode_FromString"));
    g_py.PyUnicode_AsUTF8      = reinterpret_cast<decltype(g_py.PyUnicode_AsUTF8)>(load("PyUnicode_AsUTF8"));
    g_py.PyObject_Str          = reinterpret_cast<decltype(g_py.PyObject_Str)>(load("PyObject_Str"));
    g_py.PyRun_SimpleString    = reinterpret_cast<decltype(g_py.PyRun_SimpleString)>(load("PyRun_SimpleString"));
    g_py.PyRun_String          = reinterpret_cast<decltype(g_py.PyRun_String)>(load("PyRun_String"));
    g_py.PyDict_SetItemString  = reinterpret_cast<decltype(g_py.PyDict_SetItemString)>(load("PyDict_SetItemString"));
    g_py.PyErr_Print           = reinterpret_cast<decltype(g_py.PyErr_Print)>(load("PyErr_Print"));
    g_py.PyErr_Occurred        = reinterpret_cast<decltype(g_py.PyErr_Occurred)>(load("PyErr_Occurred"));
    g_py.Py_DecRef             = reinterpret_cast<decltype(g_py.Py_DecRef)>(load("Py_DecRef"));

    if (!g_py.Py_InitializeEx || !g_py.PyImport_ImportModule) return false;
    if (!g_py.Py_IsInitialized()) {
        g_py.Py_InitializeEx(0);
    }
    initialized_ = true;
    return true;
#else
    return false;
#endif
}

bool PythonBridge::import_module(const std::string& module_name) {
    if (!initialize()) return false;
    std::string cmd = "import " + module_name;
    return g_py.PyRun_SimpleString(cmd.c_str()) == 0;
}

flawn::rt::NaNVal PythonBridge::call_module_fn(
    const std::string& module_name,
    const std::string& fn_name,
    const std::vector<flawn::rt::NaNVal>& args)
{
    if (!initialize()) return flawn::rt::NaNVal::null();
    PyObject* mod = g_py.PyImport_ImportModule(module_name.c_str());
    if (!mod) {
        if (g_py.PyErr_Occurred()) g_py.PyErr_Print();
        return flawn::rt::NaNVal::null();
    }
    PyObject* func = g_py.PyObject_GetAttrString(mod, fn_name.c_str());
    if (!func) {
        if (g_py.PyErr_Occurred()) g_py.PyErr_Print();
        if (g_py.Py_DecRef) g_py.Py_DecRef(mod);
        return flawn::rt::NaNVal::null();
    }

    PyObject* tuple = g_py.PyTuple_New(static_cast<Py_ssize_t>(args.size()));
    for (size_t i = 0; i < args.size(); ++i) {
        PyObject* item = nullptr;
        if (args[i].is_int()) {
            item = g_py.PyLong_FromLongLong(args[i].as_i64());
        } else if (args[i].is_double()) {
            item = g_py.PyFloat_FromDouble(args[i].as_double());
        } else if (args[i].is_ptr()) {
            const char* s = static_cast<const char*>(args[i].as_ptr());
            item = g_py.PyUnicode_FromString(s ? s : "");
        } else {
            item = g_py.PyLong_FromLongLong(0);
        }
        g_py.PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(i), item);
    }

    PyObject* res = g_py.PyObject_CallObject(func, tuple);
    if (g_py.Py_DecRef) {
        g_py.Py_DecRef(tuple);
        g_py.Py_DecRef(func);
        g_py.Py_DecRef(mod);
    }
    if (!res) {
        if (g_py.PyErr_Occurred()) g_py.PyErr_Print();
        return flawn::rt::NaNVal::null();
    }

    double d = g_py.PyFloat_AsDouble(res);
    if (g_py.PyErr_Occurred()) {
        g_py.PyErr_Print();
        if (g_py.Py_DecRef) g_py.Py_DecRef(res);
        return flawn::rt::NaNVal::null();
    }
    if (g_py.Py_DecRef) g_py.Py_DecRef(res);
    return flawn::rt::NaNVal::from_double(d);
}

bool PythonBridge::exec_python_block(const std::string& py_code) {
    if (!initialize()) return false;
    return g_py.PyRun_SimpleString(py_code.c_str()) == 0;
}

std::string PythonBridge::eval_to_string(const std::string& py_expr) {
    if (!initialize()) return "";
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* global_dict = g_py.PyModule_GetDict(main_mod);
    PyObject* res = g_py.PyRun_String(py_expr.c_str(), PY_EVAL_INPUT, global_dict, global_dict);
    if (!res) {
        if (g_py.PyErr_Occurred()) g_py.PyErr_Print();
        return "";
    }
    PyObject* str_obj = g_py.PyObject_Str(res);
    std::string out;
    if (str_obj) {
        const char* utf8 = g_py.PyUnicode_AsUTF8(str_obj);
        if (utf8) out = utf8;
        if (g_py.Py_DecRef) g_py.Py_DecRef(str_obj);
    }
    if (g_py.Py_DecRef) g_py.Py_DecRef(res);
    return out;
}

double PythonBridge::eval_to_double(const std::string& py_expr) {
    if (!initialize()) return 0.0;
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* global_dict = g_py.PyModule_GetDict(main_mod);
    PyObject* res = g_py.PyRun_String(py_expr.c_str(), PY_EVAL_INPUT, global_dict, global_dict);
    if (!res) {
        if (g_py.PyErr_Occurred()) g_py.PyErr_Print();
        return 0.0;
    }
    double val = g_py.PyFloat_AsDouble(res);
    if (g_py.Py_DecRef) g_py.Py_DecRef(res);
    return val;
}

void PythonBridge::set_global_i64(const std::string& name, int64_t val) {
    if (!initialize()) return;
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* global_dict = g_py.PyModule_GetDict(main_mod);
    PyObject* py_val = g_py.PyLong_FromLongLong(val);
    g_py.PyDict_SetItemString(global_dict, name.c_str(), py_val);
    if (g_py.Py_DecRef) g_py.Py_DecRef(py_val);
}

void PythonBridge::set_global_f64(const std::string& name, double val) {
    if (!initialize()) return;
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* global_dict = g_py.PyModule_GetDict(main_mod);
    PyObject* py_val = g_py.PyFloat_FromDouble(val);
    g_py.PyDict_SetItemString(global_dict, name.c_str(), py_val);
    if (g_py.Py_DecRef) g_py.Py_DecRef(py_val);
}

PythonBridge& bridge() {
    static PythonBridge instance;
    return instance;
}

} // namespace flawn::py
