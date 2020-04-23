set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(DLHA_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/dlha)
set(MODEL_INPUT_NAME "input")
set(MODEL_DIR /home/jost/PycharmProjects/DLHA/application/model/models)
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/dlha)

# Output directories for regular and quantized outputs
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dlha)
set(QUANTIZED_BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/quantized_dlha)

add_custom_target(DLHABundleDir ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
        )

# Final Executables
# =================
# Regular
#[[add_executable(DLHABundle $<TARGET_OBJECTS:DLHABundleMain>)
set_target_properties(DLHABundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(DLHABundle ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o png)
add_dependencies(DLHABundle DLHABundleMain DLHABundleNet)]]

# Quantized
#[[add_executable(QuantizedDLHABundle $<TARGET_OBJECTS:QuantizedDLHABundleMain>)
set_target_properties(QuantizedDLHABundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(QuantizedDLHABundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha.o png)
add_dependencies(QuantizedDLHABundle QuantizedDLHABundleMain QuantizedDLHABundleNet)]]

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
        OUTPUT
        ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o
        COMMAND
        model-compiler

        -target=arm
        -mcpu=cortex-a9
        -float-abi=hard

        -verbose
        -g
        -model=${MODEL_DIR}/dlha.onnx
        -onnx-define-symbol=N,1
        -model-input=${MODEL_INPUT_NAME},int,[1,1,224,224]
        -backend=CPU
        -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
        DEPENDS
        model-compiler DLHABundleDir
)
add_custom_target(DLHABundleNet ALL DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o)

# Quantization Profile
add_custom_command(
        OUTPUT
        ${DLHA_BUNDLE_DIR}/profile.yml
        COMMAND
        image-classifier ${IMAGES}/*.png
        -g
        -verbose
#        -dump-graph
        -dump-graph-DAG=${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha_quantization_profile_graph.dot
#        -image-mode=neg128to127
#        -image-channel-order=RGB
#        -image-layout=NCHW
        -dump-profile=${DLHA_BUNDLE_DIR}/profile.yml
        -onnx-define-symbol=N,1
#        -model-input=${MODEL_INPUT_NAME},int32,[1,1,32,32]
        -m=${MODEL_DIR}/dlha.onnx
        -model-input-name=${MODEL_INPUT_NAME}
        DEPENDS
        image-classifier
)
add_custom_target(DLHABundleQuantizationProfile DEPENDS ${DLHA_BUNDLE_DIR}/profile.yml)

# Quantized Bundle
add_custom_command(
        OUTPUT
        ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha.o
        COMMAND
        model-compiler
        # Cross-Compile Options
        -target=arm
        -mcpu=cortex-a9
        -float-abi=hard

        -verbose
        -g
        -load-profile=profile.yml
#        -quantization-schema=symmetric_with_uint8
        -quantization-precision=Int8
        -dump-graph-DAG=${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha_quantized_graph.dot
        -assert-all-nodes-quantized
        -keep-original-precision-for-nodes=SoftMax,Pow,Add,Div,Sub,Mul,Min,Max,Splat,InsertTensor,Reshape,BatchedAdd,MatMul,MaxPool
        -model=${MODEL_DIR}/dlha.onnx
        -onnx-define-symbol=N,1
        -model-input=${MODEL_INPUT_NAME},int,[1,1,224,224]
        -backend=CPU
        -emit-bundle=${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
        -bundle-api=dynamic
        DEPENDS
        model-compiler DLHABundleDir
)
add_custom_target(QuantizedDLHABundleNet ALL DEPENDS ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha.o DLHABundleQuantizationProfile)

# Driver program with main function for regular bundle
#[[add_library(DLHABundleMain OBJECT main.cpp)
target_compile_options(DLHABundleMain PRIVATE -std=c++11 -g)
target_include_directories(DLHABundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(DLHABundleMain DLHABundleNet)]]

# Driver program with main function for quantized bundle
#add_library(QuantizedDLHABundleMain OBJECT main.cpp)
#target_compile_options(QuantizedDLHABundleMain PRIVATE -std=c++11 -g)
#target_include_directories(QuantizedDLHABundleMain PUBLIC ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
#add_dependencies(QuantizedDLHABundleMain QuantizedDLHABundleNet)
