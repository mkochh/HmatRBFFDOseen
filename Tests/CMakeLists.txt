set(EXECUTABLES 
  oseenHLU
  oseenHLU_more_coupled_stencil
  # csp_more_coupled_stencil
  # csp_std
  # oseenILUT
  # neumann_test
  # oseenDirect
  # estimate_cond
)

foreach(e ${EXECUTABLES})
  add_executable(${e} ${e}.cpp)
  foreach(lib ${LINK_LIBRARIES})
    target_link_libraries(${e} ${lib})    
  endforeach(lib ${LINK_LIBRARIES})

endforeach(e ${EXECUTABLES})
