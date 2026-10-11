# https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/attachments/template/CMakeLists.txt

cmake_minimum_required(VERSION 4.0)

set(PROJECT_NAME FluidSim)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

# C++20 checks may appear to be tautology, but keeping in case compilation with <CXX20

# Add option to enable/disable C++ 20 module
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

# Enable C++ module dependency scanning only if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}}/../CMake")

# Apple Silicon, unlike Intel Mac does not search automatically search homebrew as path
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
# Automatically sets Vulkan_INCLUDE_DIR
find_package(Vulkan 1.4.357.0 REQUIRED)

find_package(Eigen3 REQUIRED)

# Set up Vulkan C++ module only if enabled
if (ENABLE_CPP20_MODULE)
    # vulkan.cppm
    add_library(VulkanCppModule)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)

    target_compile_definitions(VulkanCppModule PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
    target_include_directories(VulkanCppModule PUBLIC "${Vulkan_INCLUDE_DIR}")
    target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)

    set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

    target_sources(VulkanCppModule
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            BASE_DIRS
            "${Vulkan_INCLUDE_DIR}"
            FILES
            "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
    )
    # Add the vulkan.cppm file directly as a source file
    target_sources(VulkanCppModule
            PRIVATE
            "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
    )
else()
    # Create a dummy interface library when C++ 20 module is disabled
    add_library(VulkanCppModule INTERFACE)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)
    target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
    target_compile_definitions(VulkanCppModule
            INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
endif()

# TODO: look into whether I need stb (stb_image), tinygltf

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
add_subdirectory(shaders)

set(PROJECT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_executable(${PROJECT_NAME} src/main.cpp
        include/solver.h
        src/solver.cpp
        include/mutils.h)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::cppm glfw Eigen3::Eigen)
include_directories(${PROJECT_NAME} include)

# Add compile definition if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CPP20_MODULES=1)
endif()

# Define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS to treat VK_ERROR_OUT_OF_DATE_KHR as a success code
target_compile_definitions(${PROJECT_NAME} PRIVATE "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS" )

# Slang shader
## TODO: set the slang build dir to the cmake build dir
set(SHADER_TARGET ${PROJECT_NAME}Shader)
set(SHADERS_DIR ${PROJECT_DIR}/shaders)
set (SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang")
set (ENTRY_POINTS -entry vertMain -entry fragMain)
add_custom_command(OUTPUT ${SHADERS_DIR} COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR})
add_custom_command(
        OUTPUT  ${SHADERS_DIR}/slang.spv
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${SHADERS_DIR}/slang.spv
        WORKING_DIRECTORY ${SHADERS_DIR}
        DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
        COMMENT "Compiling Slang Shaders"
        VERBATIM
)
add_custom_target(${SHADER_TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
add_dependencies(${PROJECT_NAME} ${SHADER_TARGET})

target_compile_options(${PROJECT_NAME} PRIVATE
    -fsanitize=undefined,float-divide-by-zero
    -fno-sanitize-recover=all
)

target_link_options(${PROJECT_NAME} PRIVATE
    -fsanitize=undefined,float-divide-by-zero
)

# TODO: Assets DIR