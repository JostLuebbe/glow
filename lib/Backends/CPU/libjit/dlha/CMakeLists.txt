cmake_host_system_information(RESULT Device QUERY OS_RELEASE)


if (${Device} MATCHES "4.9.0-xilinx-v2017.4")
    add_definitions(-DFPGA)
endif ()

#add_library(dlha STATIC example.cpp cc.cpp cc.h utility.cpp utility.h)

add_custom_command(
        OUTPUT  example.o
        COMMAND ${CLANG_BIN} -c ../example.cpp -o example.o
        DEPENDS cc.cpp cc.h utility.cpp utility.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)