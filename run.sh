../Temp-GAN/gan/bin/python main.py --log_dir=BigGAN3 --filterG=64 --filterD=64 --hinge=True --biggan=True

../Temp-GAN/gan/bin/python main.py --log_dir=SAGAN2 --filterG=64 --filterD=64 --hinge=True --iterD=5 --lrG=1e-4 --dcgan=True --sagan=True
../Temp-GAN/gan/bin/python main.py --log_dir=SAGAN3 --filterG=64 --filterD=64 --hinge=True --iterD=5 --lrG=1e-4 --dcgan=True --sagan=True

../Temp-GAN/gan/bin/python main.py --log_dir=SNGAN2 --iterD=5 --lrG=1e-4 --dcgan=True --sngan=True
../Temp-GAN/gan/bin/python main.py --log_dir=SNGAN3 --iterD=5 --lrG=1e-4 --dcgan=True --sngan=True

../Temp-GAN/gan/bin/python main.py --log_dir=WGAN2 --iterD=5 --lrG=1e-4 --dcgan=True
../Temp-GAN/gan/bin/python main.py --log_dir=WGAN3 --iterD=5 --lrG=1e-4 --dcgan=True

../Temp-GAN/gan/bin/python main.py --log_dir=MSL2 --iterD=5 --lrG=1e-4 --dcgan=True --msl=True
../Temp-GAN/gan/bin/python main.py --log_dir=MSL3 --iterD=5 --lrG=1e-4 --dcgan=True --msl=True
