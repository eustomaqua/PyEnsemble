.. configuration.rst


======================
Set up the environment
======================

.. .. code-block: : console
..   :linenos:


We developed `ApproxBias <https://github.com/eustomaqua/ApproxBias>`_, `FairML <https://github.com/eustomaqua/FairML>`_, and `PyFairness <https://github.com/eustomaqua/PyFairness>`_ with ``Python 3.8`` and released the code to help you reproduce our work. Note that the experimental parts must be run on the ``Ubuntu`` operating system due to FairGBM (one baseline method that we used for comparison).


Initialization
==============

Configuration via Docker
-------------------------

*(1) design an image using the* ``Dockerfile`` *file*

.. code-block:: console

  $ # docker --version
  $ # docker pull continuumio/miniconda3
  $
  $ cd ~/GitH*/FairML
  $ # touch Dockerfile
  $ # vim Dockerfile           # i  # Esc :wq
  $
  $ docker build -t fairgbm .  # <image-name>
  $ docker images              # docker images -f dangling=true
  $ docker run -it fairgbm /bin/bash


*(2) enter the image and install Miniconda3 (with root access)*

.. code-block:: console

  $ cd home
  $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $
  $ bash ./Miniconda3-latest-Linux-x86_64.sh
  Do you accept the license terms? [yes|no]
  >>> yes
  Miniconda3 will now be installed into this location:
  [/root/miniconda3] >>> /home/miniconda3
  You can undo this by running `conda init --reverse $SHELL`? [yes|no]
  [no] >>> no


*(3) modify environment variables and then make it work*

.. code-block:: console

  $ vim ~/.bashrc
  export PATH=/home/miniconda3/bin:$PATH  # added by Anaconda3 installer
  $
  $ source ~/.bashrc
  $ conda env list


*(4) create and delete an environment for reproduction (see :doc:`requirements <posts/quickstart>` )*

*(5) exit from docker and delete the image*

.. code-block:: console

  $ exit
  $ docker ps -a                  # docker container ps|list
  $ docker rm <container-id>
  $ docker image rm <image-name>  # docker rmi <image-id>


Configuration on the server
----------------------------

.. code-block:: console
  
  $ ssh hendrix
  $ srun -p gpu --pty --time=2-00:00:00 --gres gpu:0 bash
  $ module load singularity
  $ mkdir Singdocker
  $ cd Singdocker
  $ singularity build --sandbox miniconda3 docker://continuumio/miniconda3
  
  $ singularity shell --writable miniconda3   # <image-name>
  Singularity> ls && conda env list
  Singularity> conda create -n fmpar python=3.8
  Singularity> source activate fmpar
  (fmpar) Singularity> pip list
  (fmpar) Singularity> conda deactivate
  Singularity> exit
  
  $ singularity build enfair.sif miniconda3/  # <environment-name>
  $ # singularity instance list
  $ # singularity cache list -v
  $ # singularity cache clean
  $ # singularity exec enfair.sif /bin/echo Hello World!
  $ singularity shell enfair.sif              # singularity run *.sif
  
  $ rm enfair.sif
  $ yes | rm -r miniconda3
  [qgl539@hendrixgpu04fl Singdocker]$ exit
  [qgl539@hendrixgate03fl ~]$ exit
  logout


Remote connection via SSH
----------------------------------

.. Permission, access, SSH into a remote server
.. https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey
.. https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server

.. code-block:: console

  $ cd ~/.ssh && ls
  $
  $ ssh-keygen -t ed25519 -C "yjbian92@gmail.com"

  Enter file in which to save the key (/root/.ssh/id_ed25519):
  Enter passphrase (empty for no passphrase):
  Enter same passphrase again:
  
  $ cat id_ed25519.pub
  $
  $ vim ~/.ssh/config

  Host nscc
      HostName  aspire2a.nus.edu.sg
      User      yjbian
      Port      22  # 8080
      IdentityFile  ~/.ssh/id_rsa

  Host hendrix
      HostName  hendrixgate  # 03fl
      User      qgl539
      StrictHostKeyChecking  no
      CheckHostIP            no
      UserKnownHostsFile=/dev/null

  $ # cat known_hosts



Implementation
==============

Executing via Docker
-------------------------

.. code-block:: console
  
  $ docker ps -a
  $ docker cp /home/yijun/<folder> <container-id>:/home/  # copy to docker
  
  $ docker restart <container-id>
  $ docker exec -it <container-id> /bin/bash
  (base) # cd home/FairML                                 # cd root/FairML
  (base) # conda activate fmpar
  (fmpar) # ....
  (fmpar) # conda deactivate
  (base) # exit
  
  $ docker cp <container-id>:/home/<folder> /home/yijun/  # copy from docker
  $ docker stop <container-id>


Executing on the server
-------------------------

.. code-block:: console

  $ rsync -r FairML hendrix:/home/qgl539/GitH/     # copy to server
  $ ssh hendrix
  $ screen                                         # screen -r <pts-id>
  $ srun -p gpu --pty --time=23:30:00 --gres gpu:0 bash
  $ module load singularity
  $ cd Singdocker
  $ singularity run enfair.sif

  Singularity> cd ~/GitH/FairML
  Singularity> source activate ensem
  (ensem) Singularity> # executing ....
  (ensem) Singularity> conda deactivate && cd ..
  (base) Singularity> tar -czvf tmp.tar.gz FairML  # compression
  (base) Singularity> yes | rm -r FairML

  (base) Singularity> exit
  [qgl539@hendrixgpu04fl Singdocker]$ exit
  [qgl539@hendrixgate01fl ~]$ exit    # exit screen
  [qgl539@hendrixgate01fl ~]$ logout  # Connection to hendrixgate closed.
  $ rsync -r hendrix:/home/qgl539/tmp.tar.gz .     # copy from server
  $ tar -xzvf tmp.tar.gz                           # decompression
  $ rm tmp.tar.gz


Documentation
=============

.. code-block:: console

  $ cd ~/GitH*/PyFairness
  $ mkdir docs && cd docs
  $ sphinx-quickstart

  Welcome to the Sphinx 8.2.3 quickstart utility.
  > Separate source and build directories (y/n) [n]: n
  > Project name: PyEnsemble
  > Author name(s): eustomadew
  > Project release []: 0.1.0
  > Project language [en]: en

  $ make html

