# Flask server

Communicate with client and calculate forecast using data.

## Getting Started

Testing is not ready.

### Prerequisites

- Install pycharm >=2017.1.5
- Install python >=3.5.3
- Need firebase-realtime-database service.

### Installing

A step by step series of examples that tell you have to get a development env running

Steps of install.

- Install Flask >=0.12.2
```
pip3.5 install Flask==0.12.2
```

- Install python-firebase-1.2 or higher
```
pip3.5 install python-firebase==1.2
```

- Install requests >=2.18.3
```
pip3.5 install requests==2.18.3
```

- Install tensorflow >=1.2.1
```
pip3.5 install tensorflow==1.2.1
```

- Install pytest-flask >=3.2.1
```
pip3.5 install pytest-flask==3.2.1
```

- Install pandas >= 0.20.3
```
pip3.5 install pandas==0.20.3
```

- Install Cython >= 0.26
```
pip3.5 install Cython==0.26
```

- Install fbprophet >= 0.1.1
```
pip3.5 install fbprophet==0.1.1
```

- Install pystan >= 2.14
```
pip3.5 install pystan==2.14
```

### Ubuntu

- Install gcc and g++
```
apt-get install g++ gcc
```

- Install python-dev
```
apt-get install python3-dev 
```

- Install python3-tk

```
apt-get install python3-tk
```

- Install a lot of things.
```
apt-get install python-dateutil python-docutils python-feedparser python-gdata python-jinja2 python-ldap python-libxslt1 python-lxml python-mako python-mock python-openid python-psycopg2 python-psutil python-pybabel python-pychart python-pydot python-pyparsing python-reportlab python-simplejson python-tz python-unittest2 python-vatnumber python-vobject python-webdav python-werkzeug python-xlwt python-yaml python-zsi
apt-get install libxml2-dev libxslt1-dev
apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
```

And run.

```
export FLASK_APP=index.py
flask run
```

## Running the tests

This project use pytest lib.

Install pytest
```
pip3.5 install pytest
```

Run test
```
pytest ConfTest.py
```

## Deployment

Enter the ncloud server using ssh.

Move to Flask-server directory.

Make your permission as root.
```
sudo su
```

And run `./runner.sh`

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## HTML TEMPLATE

- [Projection](https://templated.co/projection)