# Captcha prediction from image

## Table of Content
  * [Overview](#overview)
  * [About](#About)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#Run)
  * [To Do](#to-do)
  * [Technologies Used](#technologies-used)
  * [Credits](#credits)

## Overview
This is a Captcha character recognition model build using PyTorch. It uses a custom deep learning model with GRU(Gated recurrent unit). The project was executed on Kaggle kernels.

## About
Computer Vision has a wide variety of topics and OCR(optical character recognition) is one of them.It is a widespread technology to recognise text inside images, such as scanned documents and photos. OCR technology is used to convert virtually any kind of images containing written text (typed, handwritten or printed) into machine-readable text data.
GRU was used to read the image channel wise and hence helps in reading the characters acccurately.
One of the most widely uses of OCR are Vehicle number plate recognition, extraction of data from Resumes and Invoices.
The most useful task of OCR is to convert old hard paper documents to digital Docs.

## Motivation
The actual use of LSTMs in computer vision excited me to work on this project. 

## Technical Aspect
1. Training a deep learning model using Pytoch.
      - model: custom + GRU
      - data: [here](https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip)
      - evaluation metrics: Accuracy
2. Building a web app using Flask (will later host it on heroku too. )
   
## Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

## Run
To make life easier I've compiled everything into a notebook, so If you're interested in running the app simply run the notebook on any GPU provied platforms(kaggle suggested)

## To Do
2. Add a better vizualization chart to display the predictions.
3. deploy the Flask app on heroku with saved_model.pkl

## Bug / Feature Request
If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/zues1234/Melanoma-deeplearning/issues/new). Please include sample queries and their corresponding results.

## Technologies used
  Major frameworks & Libraries 
  * [PyTorch](https://pytorch.org/)
  * [Flask](https://flask.palletsprojects.com/en/1.1.x/)
  * [Albumentation](https://albumentations.ai/)
  * [Numpy | Pandas]
