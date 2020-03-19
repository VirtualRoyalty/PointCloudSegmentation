#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from semantic.modules.user import *


def get_user(dataset, model):
  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", dataset)
  print("model", model)
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file from %s" % model)
    ARCH = yaml.safe_load(open(model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % model)
    DATA = yaml.safe_load(open(model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # does model folder exist?
  if os.path.isdir(model):
    print("model folder exists! Using model from %s" % (model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()

  # create user and infer dataset
  user = User(ARCH, DATA, dataset, "", model)
  return user


#user = get_user()  
#predict = user.infer()