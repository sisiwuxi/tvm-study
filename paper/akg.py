# AKG: automatic kernel generation for NPUs

# ---------------------------------------------- #
# Fig3(a): The DSL exprssion of a running example
# A tyical fused pattern:
#       a 2D convolution, feature maps A using kernels B
#       constant bias addition
A = te.placeholder((H,W), name="A")
A = te.compute((,),lambda h,w : A[h,w] + bias, name="A")
B = te.placeholder((KH,KW), name="B")
kh = te.reduce_axis((0,KH), "kh")
kw = te.reduce_axis((0,KW), "kw")
C = te.compute((H-KH+1,W-KW+1), lambda h,w):
    te.sum(A[h+kh,w+kw]*B[kh,kw]),axis=kh,kw), name="C")
C = te.compute((,), lambda h,w: abs(C[h,w]), name="C")
C = te.compute((,), lambda h,w: ReLU(C[h,w]), name="C")

# Fig3(b): The initial schdule tree
Domain
    Sequence
        Filter{S0(h,w)}
            Band{S0->(h,w)}
        Filter{S1(h,w);S2(h,w,kh,kw)}
            Band{S1->(h,w), S2->(h,w)}
                Sequence
                    Filter{S1(h,w)}
                    Filter{S2(h,w,kh,kw)}
                        Band{S2->(kh,kw)}
        Filter{S3(h,w)}
            Band{S3->(h,w)}    
        Filter{S4(h,w)}
            Band{S4->(h,w)}  

# Fig3(c): The schedule tree after polyhedral scheduling
Domain
    Sequence
        Filter{S0(h,w)}
            Band{S0->(h,w)}
        Filter{S1(h,w); S2(h,w,kh,kw); S3(h,w); S4(h,w)}
            Band{S1->(h,w), S2->(h,w), S3->(h,w), S4->(h,w)}
                Sequence
                    Filter{S1(h,w)}
                    Filter{S2(h,w,kh,kw)}
                        Band{S2->(kh,kw)}
                    Filter{S3(h,w)}
                    Filter{S4(h,w)}

# Fig3(d): Tiling live-out iteration space
Domain
    Sequence
        Filter{S0(h,w)} # an intermediate iteration space
            Band{S0->(h,w)}
        Filter{S1(h,w); S2(h,w,kh,kw); S3(h,w); S4(h,w)} # a live-out iteration space
            Band{S1->(h/32,w/32), S2->(h/32,w/32), S3->(h/32,w/32), S4->(h/32,w/32)}
                Band{S1(h,w)->(h,w), S2(h,w,kh,kw)->(h,w), S3(h,w)->(h,w), S4(h,w)->(h,w)}
                    Sequence
                        Filter{S1(h,w)}
                        Filter{S2(h,w,kh,kw)}
                            Band{S2->(kh,kw)}
                        Filter{S3(h,w)}
                        Filter{S4(h,w)}

# Fig3(e): Post-tiling fusion using an extension node
Domain
    Sequence
        Filter{S0(h,w)}
            Mark{"skipped"} # The nodes below will not be scanned by code generator.
                Band{S0->(h,w)}
        Filter{S1(h,w); S2(h,w,kh,kw); S3(h,w); S4(h,w)}
            Band{S1->(h/32,w/32), S2->(h/32,w/32), S3->(h/32,w/32), S4->(h/32,w/32)}
                Extension # introduce foreign subtree, i.e., S0, to the live-out subtree
                    Sequence
                        Filter{S0(h,w)}
                            Band{S0->(h,w)}
                        Filter{S1(h,w); S2(h,w,kh,kw); S3(h,w); S4(h,w)}
                            Band{S1->(h,w), S2->(h,w), S3->(h,w), S4->(h,w)}
                                Sequence
                                    Filter{S1(h,w)}
                                    Filter{S2(h,w,kh,kw)}
                                        Band{S2->(kh,kw)}
                                    Filter{S3(h,w)}
                                    Filter{S4(h,w)}
Extension = {(o1,o2)->S0(h,w): 32*o0<=h<=32*o0+KH+31 ^ 32*o1<=w<=32*o1+KW+31 ^ 0<=o0<=roof((H-KH+1)/32) ^ 0<=o1<=roof((W-KW+1)/32)}

# Fig3(f): Intra-tile fusion using clustering strategies
Domain
    Sequence
        Skipped Filter{S0(h,w)}
        Filter{S1(h,w); S2(h,w,kh,kw); S3(h,w); S4(h,w)}
            Band{S1->(h/32,w/32), S2->(h/32,w/32), S3->(h/32,w/32), S4->(h/32,w/32)}
                Extension
                    Sequence
                        Filter{S0(h,w)}
                            Mark{"local_UB"}
                                Band{S0->(h,w)}
                        Filter{S1(h,w);S2(h,w,kh,kw)}
                            Band{S1->(h,w), S2->(h,w)}
                                Sequence
                                    Filter{S1(h,w)}
                                    Filter{S2(h,w,kh,kw)}
                                        Band{S2->(kh,kw)}
                        Filter{S3(h,w)}
                            Mark{"local_UB"}
                                Band{S3->(h,w)}    
                        Filter{S4(h,w)}
                            Mark{"local_UB"}
                                Band{S4->(h,w)}  
Extension = {(o1,o2)->S0(h,w): 32*o0<=h<=32*o0+KH+31 ^ 32*o1<=w<=32*o1+KW+31 ^ 0<=o0<=roof((H-KH+1)/32) ^ 0<=o1<=roof((W-KW+1)/32)}

# ---------------------------------------------- #
# Fig5(a): The pseudo codes of the running example
# Initial program
for h in [0,H), w in [0,W):
    A[h,w] = A[h,w] + bias                  # S0
for h in [0,H-KH], w in [0,W-KW]:
    C[h,w] = 0                              # S1
    for kh in [0,KH), kw in [0,KW):
        C[h,w] += A[h+kh, w+kw] * B[kh, kw] # S2
for h in [0,H-KH], w in [0,W-KW]:
    C[h,w] = abs(C[h,w])                    # S3
for h in [0,H-KH], w in [0,W-KW]:
    C[h,w] = ReLU(C[h,w])                   # S4

# Fig5(b): fused before tiling
for h in [0,H), w in [0,W):
    A[h,w] = A[h,w] + bias                 # S0
for h in [0,H-KH], w in [0,W-KW]:
    C[h,w] = 0                              # S1
    for kh in [0,KH), kw in [0,KW):
        C[h,w] += A[h+kh, w+kw] * B[kh, kw] # S2
    C[h,w] = abs(C[h,w])                    # S3
    C[h,w] = ReLU(C[h,w])                   # S4

# Fig5(c): pre-tile fused, tiled, post-tile fused, and rescheduled
for x0 in [0,(H-KH)/32], x1 in [0,(W-KW)/32]:
    for x2 in [0,KH+31], x3 in [0, KW+31]:                      # S0, overlapped(exec. on UB), user buffer?
        A[32*x0+x2, 32*x1+x3] = A[32*x0+x2, 32*x1+x3] + bias    
    for x2 in [0,31], x3 in [0,31]:                             # S1, (exec. on L1)
        C[32*x0+x2, 32*x1+x3] = 0
        for kh in [0,KH), kw in [0,KW):                         # S2, (exec. on L1)
            C[32*x0+x2, 32*x1+x3] += A[32*x0+x2+kh, 32*x1+x3+kw] * B[kh, kw]
    for x2 in [0,31], x3 in [0,31]:                             # S3, (exec. on UB)
        C[32*x0+x2, 32*x1+x3] = abs(C[32*x0+x2, 32*x1+x3])
    for x2 in [0,31], x3 in [0,31]:                             # S4, (exec. on UB)
        C[32*x0+x2, 32*x1+x3] = ReLU(C[32*x0+x2, 32*x1+x3])        