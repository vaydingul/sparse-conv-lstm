# -*- coding:utf-8 -*-
# author: Xinge
# @file: log_util.py 


def save_to_log(logdir, logfile, message):

    # Open the file to append
    f = open(logdir + '/' + logfile, "a")

    # Write the message
    f.write(message + '\n')
    
    # Close the file
    f.close()
    return