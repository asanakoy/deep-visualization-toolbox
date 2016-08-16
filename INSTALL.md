$ git remote add asanakoy git@github.com:asanakoy/caffe.git
$ git fetch --all
$ git checkout --track -b deconv-deep-vis-toolbox asanakoy/deconv-deep-vis-toolbox
$ < edit Makefile.config to suit your system if not already done in Step 0 >
$ make clean
$ make -j
$ make -j pycaffe

$ sudo apt-get install python-opencv scipy python-skimage


$ git clone git@github.com:asanakoy/deep-visualization-toolbox.git
$ cd deep-visualization-toolbox

$ cp models/caffenet-yos/settings_local.template-caffenet-yos.py settings_local.py # your own settings
$ < edit settings_local.py >

$ cd models/caffenet-yos/
$ ./fetch.sh
$ cd ../..

# Run
$ ./run_toolbox.py