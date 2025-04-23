# -----------------------------------------------------------
# Dockerfile for OceanWave3D on Ubuntu 20.04
# with Harwell, SPARSKIT2 (as libskit), and system LAPACK/BLAS
# -----------------------------------------------------------

    FROM ubuntu:20.04

    # Prevent interactive install prompts
    ARG DEBIAN_FRONTEND=noninteractive
    
    # 1) Install packages, including BLAS & LAPACK
    RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        tar \
        gfortran \
        python3 \
        git \
        liblapack-dev \
        libblas-dev
    
    RUN rm -rf /var/lib/apt/lists/*
    
    WORKDIR /build
    
    # 2) Copy your tar files (Harwell, lapack) if you still need them
    COPY Harwell.tar.gz /build/Harwell.tar.gz
    COPY lapack-3.12.0.tar.gz /build/lapack-3.12.0.tar.gz
    
    # 3) Clone SPARSKIT2 and OceanWave3D repositories
    RUN git clone https://github.com/efocht/SPARSKIT2.git
    RUN git clone -b botp https://github.com/apengsigkarup/OceanWave3D-Fortran90.git
    
    # 4) Copy your custom makefiles (common.mk.docker, makefile.oceanwave3d)
    COPY common.mk.docker /build/.
    COPY makefile.oceanwave3d /build/.
    
    # 5) Unpack Harwell and LAPACK
    RUN tar -xzf Harwell.tar.gz && tar -xzf lapack-3.12.0.tar.gz
    
    # -----------------------------------------------------------
    # Build Harwell
    WORKDIR /build/Harwell
    RUN make clean || true
    
    # Create /root/lib so Harwell can place libharwell.a there
    RUN mkdir -p /root/lib
    
    # Symlink gfortran-4.4 to the actual 'gfortran'
    RUN apt-get update && apt-get install -y gfortran && \
        ln -sf /usr/bin/gfortran /usr/bin/gfortran-4.4    
    RUN make
    
    # Move libharwell.a into /usr/lib
    RUN cp /root/lib/libharwell.a /usr/lib/
    RUN ranlib /usr/lib/libharwell.a
    
    # -----------------------------------------------------------
    # Build SPARSKIT2
    WORKDIR /build/SPARSKIT2
    RUN make clean || true
    RUN make
    
    # Check if SPARSKIT2 produces libsparskit2.a and rename it to libskit.a
    RUN test -f libsparskit2.a && mv libsparskit2.a libskit.a || true
    
    # Place libskit.a in /usr/lib
    RUN cp libskit.a /usr/lib/
    RUN ranlib /usr/lib/libskit.a
    
    # -----------------------------------------------------------
    # Build OceanWave3D
    WORKDIR /build/OceanWave3D-Fortran90
    
    # Copy the docker-specific build config
    RUN cp /build/common.mk.docker common.mk
    RUN cp /build/makefile.oceanwave3d makefile.oceanwave3d
    
    # Use the "Release" target (compiles to /build/oceanwave3d)
    RUN make -f makefile.oceanwave3d Release
    
    # Copy the final lowercase binary to /usr/local/bin with uppercase name
    RUN cp /build/oceanwave3d /usr/local/bin/OceanWave3D
    
    WORKDIR /app
    ENTRYPOINT ["/usr/local/bin/OceanWave3D"]