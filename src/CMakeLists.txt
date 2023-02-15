set(BINARY ${CMAKE_PROJECT_NAME})

set (SOURCE 
    main.cpp
)

add_executable(${BINARY} ${SOURCE})

add_library(${BINARY}_lib STATIC ${SOURCE})

target_compile_features(${BINARY} PRIVATE cxx_std_17)
target_compile_features(${BINARY}_lib PRIVATE cxx_std_17)

