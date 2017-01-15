# -*- coding: utf-8 -*-
# encoding=utf-8
dfile =open('E:\\paper\\df\\2037\\2037.txt','r')
dictionary = dfile.read().decode('utf-8').strip('\n').split('\n')
print dictionary


file =open('E:\\paper\\\split_result\\train\\all1.txt','r')
dictionary1 = file.read().strip().replace(',',' ').strip(' ').decode('utf-8').split(' ')

print 'most'
print dictionary1