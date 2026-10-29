include(GoogleTest)

add_executable(smolvla_rt_tests
  test_bundle.cpp
  test_contract.cpp
  test_control_loop.cpp
  test_cuda_device.cpp
  test_manifest.cpp
  test_runtime.cpp
  test_stats.cpp
  test_tensor.cpp)
target_link_libraries(smolvla_rt_tests PRIVATE
  smolvla_rt smolvla_rt_flags nlohmann_json::nlohmann_json GTest::gtest_main)
gtest_discover_tests(smolvla_rt_tests DISCOVERY_MODE PRE_TEST)

if(SMOLVLA_RT_WITH_ORT)
  # Reads SMOLVLA_RT_FIXTURE_DIR when it runs and is reported as skipped without it; tests/cpp_parity.sh sets it.
  add_executable(smolvla_rt_parity_tests test_parity.cpp)
  target_link_libraries(smolvla_rt_parity_tests PRIVATE smolvla_rt smolvla_rt_flags nlohmann_json::nlohmann_json
                                                        GTest::gtest_main)
  gtest_discover_tests(smolvla_rt_parity_tests DISCOVERY_MODE PRE_TEST
                       PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
endif()
