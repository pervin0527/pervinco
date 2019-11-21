Tensorflow 2.0 INSTALL GUIDE
=


2019.11.20일 기준 

Release 2.0.0 
[참고 사이트](https://github.com/tensorflow/tensorflow/releases)

## System requirements
- **pip 19.0 or later (requires manylinux2010 support)** 중요 pip 버전 안 맞으면 2.0.0다운 안 됨
- Ubuntu 16.04 or later (64-bit)
- macOS 10.12.6 (Sierra) or later (64-bit) (no GPU support)
- Windows 7 or later (64-bit) (Python 3 only)
- Raspbian 9.0 or later



## pip version 맞추기
 
```console
$ pip --version
```


만약 버전이 맞지 않는다면 아래 진행

pip version upgrade
sudo -H pip3 install --upgrade pip
sudo -H pip install --upgrade pip


```console
$ pip3 install --user --upgrade tensorflow  # install in $HOME
$ pip3 install --user --upgrade tensorflow-gpu  # install in $HOME

If version 2.0 wasn't installed

$ pip3 install --upgrade tensorflow==2.0.0
$ pip3 install --upgrade tensorflow-gpu==2.0.0
```