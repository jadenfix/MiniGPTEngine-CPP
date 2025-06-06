# CMake generated Testfile for 
# Source directory: /Users/jadenfix/CascadeProjects/lightgpt
# Build directory: /Users/jadenfix/CascadeProjects/lightgpt/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_model_loader "/Users/jadenfix/CascadeProjects/lightgpt/build/test_model_loader")
set_tests_properties(test_model_loader PROPERTIES  _BACKTRACE_TRIPLES "/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;157;add_test;/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;0;")
add_test(test_tensor "/Users/jadenfix/CascadeProjects/lightgpt/build/test_tensor")
set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;157;add_test;/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;0;")
add_test(test_fixes "/Users/jadenfix/CascadeProjects/lightgpt/build/test_fixes")
set_tests_properties(test_fixes PROPERTIES  _BACKTRACE_TRIPLES "/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;157;add_test;/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;0;")
add_test(test_transformer "/Users/jadenfix/CascadeProjects/lightgpt/build/test_transformer")
set_tests_properties(test_transformer PROPERTIES  _BACKTRACE_TRIPLES "/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;157;add_test;/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;0;")
add_test(test_tokenizer "/Users/jadenfix/CascadeProjects/lightgpt/build/test_tokenizer")
set_tests_properties(test_tokenizer PROPERTIES  _BACKTRACE_TRIPLES "/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;157;add_test;/Users/jadenfix/CascadeProjects/lightgpt/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("examples")
