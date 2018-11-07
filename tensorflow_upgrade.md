## Tensorflow upgrade

Current Version: 1.4.1

Versions Tested:
 - 1.5.0
 - 1.6.0
 - 1.7.0
 - 1.8.0 Upgrading to 1.8.0 breaks no functionality and produces no error when running with Travis CI. Importing data models also do not seem to produce any problems.
 - 1.9.0 & 1.10.1 These versions gave me an _ImportError: dlopen: cannot load any more object with static TLS_ Looking online, this seemed to be a bug associated with these versions. On upgrading to 1.12.0, this issue no longer persisted. (Travis CI log: [1.9.0](https://travis-ci.com/c0derlint/Fabrik/builds/90459224), [1.10.1](https://travis-ci.com/c0derlint/Fabrik/builds/90600645))
 - 1.12.0 (latest) Travis CI build successfully ran on this version of tensorflow. Upon importing and exporting models, there seem to be no errors.

 ![Tensorflow 1.12.0 import](https://i.imgur.com/8CZpF5q.png)