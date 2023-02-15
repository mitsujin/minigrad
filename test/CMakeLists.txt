set (BINARY ${CMAKE_PROJECT_NAME}_test)
set (SOURCE 
    main.cpp
    tensor_storage_tests.cpp
)

add_executable(${BINARY} ${SOURCE})
add_test(NAME ${BINARY} COMMAND ${BINARY})

target_link_libraries(${BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest ${TORCH_LIBRARIES})
target_compile_features(${BINARY} PRIVATE cxx_std_17)
