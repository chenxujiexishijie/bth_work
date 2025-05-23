# Pickle Stability and Correctness Test Suite

This project investigates the stability and correctness of Python's `pickle` module.  It aims to determine if the same input object consistently produces the same serialized byte stream (and thus the same hash) across different environments, Python versions, and under various conditions like handling floating-point numbers and recursive data structures.

## Project Structure and Files

### Core Scripts

* **`final_work.py`**:
    * This script is the primary tool for generating pickled representations of various Python objects and calculating their SHA256 hashes across different pickle protocols.
    * It systematically tests basic data types, container types, complex nested structures, recursive objects, custom objects, and fuzzed data inputs. 
    * The output is a JSON file containing the description of the test case and the hashes generated for each tested protocol in a specific Python environment.
* **`compair.py`**:
    * This script compares two JSON output files generated by `final_work.py` from different environments.
    * It highlights differences in hashes for the same test case and protocol, indicating potential instabilities in the pickling process.

### Output Files and Reports

* **`pickle_hashes_py36_win32_seed42.json`**: Output from `final_work.py` run on Python 3.6 (32-bit) on Windows, using seed 42 for fuzzing. This file was generated *after* the initial report was written, using the modified `final_work.py`.
* **`pickle_hashes_py311_darwin_seed42.json`**: Output from `final_work.py` run on Python 3.11 on macOS (Darwin), using seed 42 for fuzzing. This file was generated *after* the initial report was written, using the modified `final_work.py`.
* **`pickle_hashes_py311_win32_seed42.json`**: Output from `final_work.py` run on Python 3.11 (32-bit) on Windows, using seed 42 for fuzzing. This file was generated *after* the initial report was written, using the modified `final_work.py`.
* **`report_output.txt`**: This text file contains the comparison results generated by `compair.py` using output files from the version of `final_work.py` that was current *at the time of writing the project report*.
* **`after_report_modify_output.txt`**: This text file contains comparison results generated by `compair.py` using the `pickle_hashes_*.json` files listed above. These files were produced by a version of `final_work.py` that was modified *after* the project report was completed.

### Supporting Files

* **`input.json`**: This file was an earlier attempt to store expected input representations (`input_repr`) and their corresponding hashes. The `input_repr` field from this file was a useful reference for understanding the intended input for each test case.


## Important Note on Code Modification and Report Discrepancies

After the initial project report was written, the `final_work.py` script was modified. The primary reason for this modification was to enhance the output JSON files by including a representation of the **input object** itself (`input_repr`) for each test case. This was intended to provide better context when analyzing the generated hashes, as the original script only outputted the hashes and descriptions.

This modification, while not altering the fundamental input data being pickled, led to **minor variations** in the structure of the output JSON data and, in some specific cases related to object representation or internal script state, potentially subtle differences in the calculated hashes compared to those generated by the script version used for the original report.

Consequently:

* The `pickle_hashes_*.json` files present in this repository reflect the output of the *modified* `final_work.py`.
* The `after_report_modify_output.txt` file shows the comparison results based on these newer `pickle_hashes_*.json` files.
* The `report_output.txt` file reflects the comparison results that were used *during the writing of the final project report*, based on the older version of `final_work.py`.

The report was **not updated** with the results from the modified script due to time constraints following the report's completion. The core findings regarding pickle stability across different Python versions and operating systems, as detailed in the report, were based on the `report_output.txt` data. The later modifications were primarily for enhancing the traceability and debuggability of the test outputs, rather than altering the fundamental test logic or the inputs being pickled.

## How to Use

1.  **Generate Hashes**:
    * Run `final_work.py` in different Python environments (e.g., different Python versions, different operating systems).
    * This will produce a JSON file (e.g., `pickle_hashes_py<version>_<platform>_seed<seed>.json`) containing the hashes for various test cases and pickle protocols.
2.  **Compare Results**:
    * Use `compair.py` to compare two of the generated JSON files.
    * Example: `python compair.py pickle_hashes_py36_win32_seed42.json pickle_hashes_py311_win32_seed42.json`
    * The script will print out any discrepancies found between the two files.

## Interpreting Results

* **MISSING**: Indicates a protocol or test case exists in one file but not the other.
* **MISMATCH**: Indicates that the hash for a specific test case and protocol differs between the two files. This is a key indicator of potential pickle instability. 
* The `after_report_modify_output.txt` shows that there are differences in hashes when comparing Python 3.6 (Win32) with Python 3.11 (Win32), and also when comparing Python 3.11 (Darwin) with Python 3.11 (Win32).  These differences are concentrated in specific test cases, particularly involving sets, frozensets, and certain custom or fuzzed objects.


# Pickle 稳定性与正确性测试套件

本项目旨在研究 Python `pickle` 模块的稳定性和正确性。  其目标是确定相同的输入对象是否能在不同环境、不同 Python 版本以及处理浮点数和递归数据结构等各种条件下，始终产生相同的序列化字节流（并因此产生相同的哈希值）。 

## 项目结构与文件

###核心脚本

* **`final_work.py`**:
    * 该脚本是用于生成各种 Python 对象的 pickle 表示，并在不同 pickle 协议下计算其 SHA256 哈希值的主要工具。
    * 它系统地测试了基本数据类型、容器类型、复杂嵌套结构、递归对象、自定义对象以及模糊测试数据输入。
    * 输出是一个 JSON 文件，包含测试用例的描述以及在特定 Python 环境中为每个测试协议生成的哈希值。
* **`compair.py`**:
    * 该脚本用于比较由 `final_work.py` 在不同环境下生成的两个 JSON 输出文件。
    * 它会高亮显示相同测试用例和协议下哈希值的差异，从而指出 pickling 过程中潜在的不稳定性。

### 输出文件与报告

* **`pickle_hashes_py36_win32_seed42.json`**: `final_work.py` 在 Python 3.6 (32位) Windows 环境下运行的输出，模糊测试使用种子 42。该文件是在初步报告编写完成*之后*，使用修改后的 `final_work.py` 生成的。
* **`pickle_hashes_py311_darwin_seed42.json`**: `final_work.py` 在 Python 3.11 macOS (Darwin) 环境下运行的输出，模糊测试使用种子 42。该文件是在初步报告编写完成*之后*，使用修改后的 `final_work.py` 生成的。
* **`pickle_hashes_py311_win32_seed42.json`**: `final_work.py` 在 Python 3.11 (32位) Windows 环境下运行的输出，模糊测试使用种子 42。该文件是在初步报告编写完成*之后*，使用修改后的 `final_work.py` 生成的。
* **`report_output.txt`**: 此文本文件包含由 `compair.py` 生成的比较结果，其使用的是*在撰写项目报告时* `final_work.py` 当前版本的输出文件。 
* **`after_report_modify_output.txt`**: 此文本文件包含由 `compair.py` 生成的比较结果，其使用的是上面列出的 `pickle_hashes_*.json` 文件。这些文件是由项目报告完成*之后*修改过的 `final_work.py` 版本生成的。

### 支持文件

* **`input.json`**: 此文件是早期用于存储预期输入表示 (`input_repr`) 及其对应哈希值的尝试。该文件中的 `input_repr` 字段为了解每个测试用例的预期输入提供了有用的参考。
* **`BTH_104_Project.docx`**: 此文档概述了项目需求和目标（仅供了解项目背景）。

## 关于代码修改与报告差异的重要说明

在初步项目报告撰写完成后，对 `final_work.py` 脚本进行了修改。此次修改的主要目的是通过在每个测试用例的输出 JSON 文件中包含**输入对象**本身的表示 (`input_repr`) 来增强输出。这样做是为了在分析生成的哈希值时提供更好的上下文，因为原始脚本仅输出了哈希值和描述。

虽然此修改并未改变被 pickle 的核心输入数据，但它导致了输出 JSON 数据结构的**微小变动**，并且在某些与对象表示或脚本内部状态相关的特定情况下，可能导致计算出的哈希值与原始报告所用脚本版本生成的哈希值存在细微差异。

因此：

* 本仓库中存在的 `pickle_hashes_*.json` 文件反映的是*修改后*的 `final_work.py` 的输出。
* `after_report_modify_output.txt` 文件显示的是基于这些较新的 `pickle_hashes_*.json` 文件的比较结果。 
* `report_output.txt` 文件反映的是*在撰写最终项目报告期间*所使用的比较结果，基于旧版本的 `final_work.py`。 

由于报告完成后时间紧张，报告**未更新**为使用修改后脚本的结果。报告中详述的关于 pickle 在不同 Python 版本和操作系统之间稳定性的核心发现，是基于 `report_output.txt` 的数据。后续的修改主要是为了增强测试输出的可追溯性和可调试性，而不是改变基本的测试逻辑或被 pickle 的输入。

## 如何使用

1.  **生成哈希值**:
    * 在不同的 Python 环境（例如，不同的 Python 版本、不同的操作系统）中运行 `final_work.py`。
    * 这将生成一个 JSON 文件（例如，`pickle_hashes_py<version>_<platform>_seed<seed>.json`），其中包含各种测试用例和 pickle 协议的哈希值。
2.  **比较结果**:
    * 使用 `compair.py` 比较两个生成的 JSON 文件。 
    * 示例: `python compair.py pickle_hashes_py36_win32_seed42.json pickle_hashes_py311_win32_seed42.json`
    * 该脚本将打印出在两个文件之间发现的任何差异。

## 结果解读

* **MISSING**: 表示某个协议或测试用例存在于一个文件中，但不存在于另一个文件中。
* **MISMATCH**: 表示特定测试用例和协议的哈希值在两个文件之间存在差异。这是 pickle 不稳定性的一个关键指标。
* `after_report_modify_output.txt` 文件显示，在比较 Python 3.6 (Win32) 与 Python 3.11 (Win32) ，以及比较 Python 3.11 (Darwin) 与 Python 3.11 (Win32) 时存在哈希值差异 [cite: 118]。这些差异集中在特定的测试用例中，特别是涉及集合 (set)、不可变集合 (frozenset) 以及某些自定义或模糊测试对象的情况。
