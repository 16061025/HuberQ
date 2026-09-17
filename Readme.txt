%%%%%%%%%%%%%%%%%%%File list%%%%%%%%%%%%%%%%%%%%%%%%%%
1. Readme.txt
2. Code folder <TableExp>
3. Code folder <DeepExp>


%%%%%%%%%%%%%%%%%%%Experiment environments%%%%%%%%%%%%%%%%%%%%%%%%%%
1.Compute resource:
-- Ubantu: 20.04.6 LTS
-- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
-- GPU: RTX 3090 * 8
-- NVIDIA-SMI: 560.35.05
-- CUDA: 12.6

2.Software resouce
-- Platform: PyCharm2025.1
-- Language: Python


3.Package for Tabular experiment
Package         Version
--------------- -----------
_libgcc_mutex             0.1         
_openmp_mutex             5.1         
abseil-cpp                20211102.0  
absl-py                   1.3.0       
aiohttp                   3.8.3       
aiosignal                 1.2.0       
async-timeout             4.0.2       
attrs                     22.1.0      
blas                      1.0         
blinker                   1.4         
brotlipy                  0.7.0       
bzip2                     1.0.8       
c-ares                    1.19.0      
ca-certificates           2025.8.3    
cachetools                4.2.2       
certifi                   2025.8.3    
cffi                      1.15.1      
charset-normalizer        2.0.4       
click                     8.0.4       
cloudpickle               2.2.1       
cmake                     3.25.0      
cryptography              39.0.1      
cuda-cudart               11.7.99     
cuda-cupti                11.7.101    
cuda-libraries            11.7.1      
cuda-nvrtc                11.7.99     
cuda-nvtx                 11.7.91     
cuda-runtime              11.7.1      
cycler                    0.12.1      
dm-control                1.0.7       
dm-env                    1.6         
dm-tree                   0.1.8       
ffmpeg                    4.3         
filelock                  3.9.0       
fonttools                 4.60.1      
freetype                  2.12.1      
frozenlist                1.3.3       
giflib                    5.2.1       
glfw                      2.5.9       
gmp                       6.2.1       
gmpy2                     2.1.2       
gnutls                    3.6.15      
google-auth               2.6.0       
google-auth-oauthlib      0.5.2       
grpc-cpp                  1.48.2      
grpcio                    1.48.2      
gym                       0.21.0      
gym-notices               0.0.8       
idna                      3.4         
imageio                   2.28.1      
imageio-ffmpeg            0.4.8       
importlib-metadata        6.0.0       
intel-openmp              2021.4.0    
jinja2                    3.1.2       
joblib                    1.5.1       
jpeg                      9e          
kiwisolver                1.4.7       
kornia                    0.6.12      
labmaze                   1.0.6       
lame                      3.100       
lcms2                     2.12        
ld_impl_linux-64          2.38        
lerc                      3.0         
libcublas                 11.10.3.66  
libcufft                  10.7.2.124  
libcufile                 1.6.1.9     
libcurand                 10.3.2.106  
libcusolver               11.4.0.1    
libcusparse               11.7.4.91   
libdeflate                1.17        
libffi                    3.3         
libgcc-ng                 11.2.0      
libgomp                   11.2.0      
libiconv                  1.16        
libidn2                   2.3.2       
libnpp                    11.7.4.75   
libnvjpeg                 11.8.0.2    
libpng                    1.6.39      
libprotobuf               3.20.3      
libstdcxx-ng              13.2.0      
libtasn1                  4.19.0      
libtiff                   4.5.0       
libunistring              0.9.10      
libwebp                   1.2.4       
libwebp-base              1.2.4       
lit                       15.0.7      
lxml                      4.9.2       
lz4-c                     1.9.4       
markdown                  3.4.1       
markupsafe                2.1.1       
matplotlib                3.5.1       
mkl                       2021.4.0    
mkl-service               2.4.0       
mkl_fft                   1.3.1       
mkl_random                1.2.2       
mpc                       1.1.0       
mpfr                      4.0.2       
mpmath                    1.2.1       
mujoco                    2.3.5       
multidict                 6.0.2       
ncurses                   6.4         
nettle                    3.7.3       
networkx                  2.8.4       
numpy                     1.22.3      
numpy-base                1.22.3      
oauthlib                  3.2.2       
opencv-python             4.12.0.88   
openh264                  2.1.1       
openssl                   1.1.1t      
packaging                 23.1        
pillow                    9.4.0       
pip                       23.0.1      
protobuf                  3.20.3      
psutil                    5.9.5       
pyasn1                    0.4.8       
pyasn1-modules            0.2.8       
pycparser                 2.21        
pyjwt                     2.4.0       
pyopengl                  3.1.6       
pyopenssl                 23.0.0      
pyparsing                 2.4.7       
pysocks                   1.7.1       
python                    3.9.0       
python-dateutil           2.9.0.post0 
re2                       2022.04.01  
readline                  8.2         
requests                  2.29.0      
requests-oauthlib         1.3.0       
rsa                       4.7.2       
scipy                     1.10.1      
setuptools                66.0.0      
six                       1.16.0      
sqlite                    3.41.2      
sympy                     1.11.1      
tensorboard               2.12.1      
tensorboard-data-server   0.7.0       
tensorboard-plugin-wit    1.8.1       
termcolor                 1.1.0       
tk                        8.6.12      
torch                     2.0.1+cu118 
torchaudio                2.0.2+cu118 
torchvision               0.15.2+cu118
tqdm                      4.65.0      
triton                    2.0.0       
typing_extensions         4.5.0       
tzdata                    2023c       
urllib3                   1.26.15     
werkzeug                  2.2.3       
wheel                     0.38.4      
xz                        5.4.2       
yarl                      1.8.1       
zipp                      3.11.0      
zlib                      1.2.13      
zstd                      1.5.5       

4.Package for deep experiment
Package                  Version
------------------------ -----------
_libgcc_mutex             0.1          
_openmp_mutex             5.1          
absl-py                   2.1.0        
ca-certificates           2025.10.5    
cachetools                5.5.2        
certifi                   2024.8.30    
cffi                      1.15.1       
charset-normalizer        3.4.4        
cloudpickle               2.2.1        
cycler                    0.11.0       
cython                    0.29.36      
farama-notifications      0.0.4        
fonttools                 4.38.0       
future                    1.0.0        
glew                      2.2.0        
glfw                      2.10.0       
glfw3                     3.2.1        
google-auth               2.41.1       
google-auth-oauthlib      0.4.6        
grpcio                    1.62.3       
gym                       0.26.2       
gym-games                 2.0.0        
gym-notices               0.1.0        
gymnasium                 0.28.1       
icu                       73.2         
idna                      3.10         
imageio                   2.31.2       
importlib-metadata        6.7.0        
jax-jumpy                 1.0.0        
kernel-headers_linux-ppc64le 3.10.0    
kiwisolver                1.4.5        
ld_impl_linux-64          2.44         
libdrm                    2.4.125      
libdrm-cos6-x86_64        2.4.65       
libdrm-cos7-ppc64le       2.4.97       
libegl                    1.7.0        
libexpat                  2.7.1        
libffi                    3.4.4        
libgcc                    15.2.0       
libgcc-ng                 15.2.0       
libgcrypt                 1.11.1       
libgcrypt-devel           1.11.1       
libgcrypt-lib             1.11.1       
libgcrypt-tools           1.11.1       
libgl                     1.7.0        
libglu                    9.0.3        
libglvnd                  1.7.0        
libglvnd-cos7-ppc64le     1.0.1        
libglvnd-glx-cos7-ppc64le 1.0.1        
libglx                    1.7.0        
libgomp                   15.2.0       
libgpg-error              1.55         
libllvm20                 20.1.8       
libnsl                    2.0.1        
libpciaccess              0.18         
libstdcxx                 15.2.0       
libstdcxx-ng              15.2.0       
libx11-common-cos6-x86_64 1.6.4        
libx11-cos6-x86_64        1.6.4        
libxcb                    1.17.0       
libxml2                   2.13.8       
libzlib                   1.3.1        
lockfile                  0.12.2       
markdown                  3.4.4        
markupsafe                2.1.5        
matplotlib                3.5.2        
mesa-khr-devel-cos7-ppc64le 18.3.4     
mesa-libgl-cos6-x86_64    11.0.7       
mesa-libgl-cos7-ppc64le   18.3.4       
mesa-libgl-devel-cos7-ppc64le 18.3.4   
mesa-libglapi-cos7-ppc64le 18.3.4      
mesalib                   25.0.5       
minatar                   1.0.15       
mujoco-py                 1.50.1.68    
ncurses                   6.5          
numpy                     1.21.0       
oauthlib                  3.2.2        
opencv-python             4.8.1.78     
openssl                   3.5.4        
osmesa                    12.2.2.dev   
packaging                 24.0         
pandas                    1.3.5        
patchelf                  0.17.2       
pillow                    9.5.0        
pip                       22.3.1       
ple                       0.0.1        
protobuf                  3.20.3       
psutil                    5.9.0        
pthread-stubs             0.3          
pyarrow                   12.0.1       
pyasn1                    0.5.1        
pyasn1-modules            0.3.0        
pycparser                 2.21         
pygame                    2.5.2        
pyglet                    1.5.0        
pyparsing                 3.1.4        
python                    3.7.12      
python-dateutil           2.9.0.post0  
pytz                      2025.2       
readline                  8.3          
requests                  2.31.0       
requests-oauthlib         2.0.0        
rsa                       4.9.1        
scipy                     1.7.3        
seaborn                   0.11.2       
setuptools                65.6.3       
six                       1.17.0       
spirv-tools               2025.4       
sqlite                    3.50.2       
swig                      4.3.1.post0  
sysroot_linux-ppc64le     2.17         
tensorboard               2.11.2       
tensorboard-data-server   0.6.1        
tensorboard-plugin-wit    1.8.1        
tk                        8.6.15       
torch                     1.13.1+cu117 
torchaudio                0.13.1+cu117 
torchvision               0.14.1+cu117 
typing-extensions         4.7.1        
urllib3                   2.0.7        
werkzeug                  2.2.3        
wheel                     0.38.4       
xorg-libx11               1.8.12       
xorg-libxau               1.0.12       
xorg-libxdamage           1.1.6        
xorg-libxdmcp             1.1.5        
xorg-libxext              1.3.6        
xorg-libxfixes            6.0.2        
xorg-libxrandr            1.5.4        
xorg-libxrender           0.9.12       
xorg-libxshmfence         1.3.3        
xorg-libxxf86vm           1.1.6        
xorg-xorgproto            2024.1       
xz                        5.6.4        
zipp                      3.15.0       
zlib                      1.3.1        
zstd                      1.5.7        



%%%%%%%%%%%%%%%%%%%Code and Running%%%%%%%%%%%%%%%%%%%%%%%%%%
1.Table Experiment
The codes are in the folder <TableExp>

To run the experiment:

cd /PATH/TO/TableExp
rm -rf ./Result/*
bash run.sh

data will be saved in the foler <TableExp/Result>. 

To generate figures of tabular results:

cd /PATH/TO/TableExp
python ./Figureutils/figureHubermultiarm.py

Figures will be shown and saved in the folder <TableExp/Result>


2.Deep Experiment
The codes are in the folder <DeepExp>

To run the experiment:

cd /PATH/TO/CODE/TableExp
rm -rf ./logs/*
nohup bash ./run.sh > run.out 2>&1 &

data will be saved in the foler <DeeqExp/logs>.

To generate figures of deep setting experiment:

cd /PATH/TO/CODE/TableExp
python ./figure_Huber_result.py

Figures will be shown and saved in the folder <DeeqExp/figures>






