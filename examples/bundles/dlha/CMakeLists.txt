set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)
set(DLHA_BUNDLE_DIR ${GLOW_BINARY_DIR}/examples/bundles/dlha)
#set(RESNET50_GLOW_S3 "http://fb-glow-assets.s3.amazonaws.com/models/resnet50")
#set(RESNET50_NET_FILES predict_net.pbtxt predict_net.pb init_net.pb)
set(MODEL_INPUT_NAME "jost_model")
set(MODEL_DIR /home/sd/DLHA/application/model/models)
set(IMAGES ${GLOW_SOURCE_DIR}/tests/images/imagenet)

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
add_executable(DLHABundle $<TARGET_OBJECTS:DLHABundleMain>)
set_target_properties(DLHABundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(DLHABundle ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o png)
add_dependencies(DLHABundle DLHABundleMain DLHABundleNet)

# Quantized
add_executable(QuantizedDLHABundle $<TARGET_OBJECTS:QuantizedDLHABundleMain>)
set_target_properties(QuantizedDLHABundle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(QuantizedDLHABundle ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha.o png)
add_dependencies(QuantizedDLHABundle QuantizedDLHABundleMain QuantizedDLHABundleNet)

# Glow Bundles
# ============
# Regular Bundle
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o
  COMMAND
    model-compiler
    -g
    -model=${MODEL_DIR}/jost_model.onnx
    -onnx-define-symbol=N,1
#    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
    -backend=CPU -emit-bundle=${BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler DLHABundleDir
)
add_custom_target(DLHABundleNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/dlha.o)

# Quantization Profile
add_custom_command(
  OUTPUT
    ${DLHA_BUNDLE_DIR}/profile.yml
  COMMAND
    image-classifier ${IMAGES}/*.png
    -i=0to1
    -dump-profile=${DLHA_BUNDLE_DIR}/profile.yml
    -m=${MODEL_DIR}/jost_model.onnx
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
    -g
    -load-profile=profile.yml
    -assert-all-nodes-quantized
    -keep-original-precision-for-nodes=SoftMax
    -model=${MODEL_DIR}/jost_model.onnx
    -onnx-define-symbol=N,1
#    -model-input=${MODEL_INPUT_NAME},float,[1,3,224,224]
    -backend=CPU
    -emit-bundle=${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}
    -bundle-api=dynamic
  DEPENDS
    model-compiler DLHABundleDir
)
add_custom_target(QuantizedDLHABundleNet DEPENDS ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY}/dlha.o DLHABundleQuantizationProfile)

# Other
# =====
# Driver program with main function for regular bundle
add_library(DLHABundleMain OBJECT main.cpp)
target_compile_options(DLHABundleMain PRIVATE -std=c++11 -g)
target_include_directories(DLHABundleMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(DLHABundleMain DLHABundleNet)

# Driver program with main function for quantized bundle
add_library(QuantizedDLHABundleMain OBJECT main.cpp)
target_compile_options(QuantizedDLHABundleMain PRIVATE -std=c++11 -g)
target_include_directories(QuantizedDLHABundleMain PUBLIC ${QUANTIZED_BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(QuantizedDLHABundleMain QuantizedDLHABundleNet)

# Network structure and weight files
#foreach(file ${RESNET50_NET_FILES})
#    add_custom_command(
#    OUTPUT
#      ${file}
#    COMMAND
#      wget
#    ARGS
#      "${RESNET50_GLOW_S3}/${file}" -P ${RESNET50_BUNDLE_DIR}/resnet50 -nc
#)
#endforeach()
#add_custom_target(ResNet50BundleNetFiles DEPENDS ${RESNET50_NET_FILES})