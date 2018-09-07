# MNIST classification by TensorFlow #

- [MNIST For ML Beginners](https://www.tensorflow.org/tutorials/mnist/beginners/)
- [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

### Requirement ###

- Python >=2.7 or >=3.4
  - TensorFlow >=1.0
- Node >=6.9


### How to run ###

    1. run `npm install` to download necessary js files
    2. if use nginx, add patterns:

            location /mnist/ {
                rewrite /mnist/(.*) /$1 break;
                proxy_pass          http://127.0.0.1:5000;
                proxy_set_header    Cookie $http_cookie;
            }

            location /static/ {
                root {project path}/tensorflow-mnist;
                index index.html;
            }
<<<<<<< HEAD
=======
			
>>>>>>> 7b7e99b44422c6222fc26fc1be875aceb50455fe
        if do not use nginx, please change main.py:
            @app.route('/api', methods=['POST'])  ==>  @app.route('/mnist/api', methods=['POST'])
            @app.route('/') ==> @app.route('/mnist')
        
