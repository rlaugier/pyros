# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:43:57 2017

@author: echo
"""
# Python standard library imports

# Third-party imports

import numpy as np

import astropy.units as u
from astropy.table import Table
import astropy.coordinates


import subprocess



import pymysql





class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False



def getxmlval(root,tag):
    ''' a simple function to retrieve a value from an XML etree object'''
    vals=[]
    for element in root.iter(tag=tag):
        vals.append(element.text)
    return vals
    
    
def gets(ra,dec,field):
    '''Just a convenient function that returns the corners of
    a field of view from its center'''
    demichamp = field/2
    racorners = np.zeros(4)
    deccorners = np.zeros(4)
    racorners[0], deccorners[0] = ra + demichamp, dec - demichamp
    racorners[1], deccorners[1] = ra + demichamp, dec + demichamp
    racorners[2], deccorners[2] = ra - demichamp, dec + demichamp
    racorners[3], deccorners[3] = ra - demichamp, dec - demichamp
    return racorners, deccorners


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump
        
                
def get_obs_info(sitename,pwd):
    '''This function retrieves all the necessary info from ROS telescopes database'''
    conn = pymysql.connect(host='cador.obs-hp.fr', user='ros', password=pwd, db='ros')
    error, idtelescope = get_db_info(conn, "telescopes", "idtelescope", "name", sitename)
    error, latitude = get_db_info(conn, "telescopes", "latitude", "name", sitename)
    error, longitude = get_db_info(conn, "telescopes", "longitude", "name", sitename)
    error, altitude = get_db_info(conn, "telescopes", "altitude", "name", sitename)
    error, sens = get_db_info(conn, "telescopes", "sens", "name", sitename)
    error, horizontype = get_db_info(conn, "telescopes", "horizontype", "name", sitename)
    error, horizondef = get_db_info(conn, "telescopes", "horizondef", "name", sitename)
    error, limdecmin = get_db_info(conn, "telescopes", "limdecmin", "name", sitename)
    error, limdecmax = get_db_info(conn, "telescopes", "limdecmax", "name", sitename)
    error, limharise = get_db_info(conn, "telescopes", "limharise", "name", sitename)
    error, limhaset = get_db_info(conn, "telescopes", "limhaset", "name", sitename)
    
    if error != 0 : 
        print("Error recovering data")
    #Turn location information into an EarthLocation object
    location = transform_location(latitude, longitude, sens, altitude)
    conn.close()
    print(horizontype, horizondef,limharise, limhaset)
    print("converting horizon")
    print (horizondef, horizontype)
    #Convert the horizon information into data Table of angle values
    myhorizon = convert_horizon(horizondef,horizontype)
    print(myhorizon)
    hadeclims = (limdecmin,limdecmax,limharise,limhaset)
    return location, myhorizon, horizontype, hadeclims,idtelescope
  
    if error != 0 : 
        print("Error recovering data")
    #Turn location information into an EarthLocation object
    location = transform_location(latitude, longitude, sens, altitude)
    conn.close()
    print(horizontype, horizondef,limharise, limhaset)
    print("converting horizon")
    print (horizondef, horizontype)
    #Convert the horizon information into data Table of angle values
    myhorizon = convert_horizon(horizondef,horizontype)
    print(myhorizon)
    hadeclims = (limdecmin,limdecmax,limharise,limhaset)
    return location, myhorizon, horizontype, hadeclims,idtelescope


def transform_location(latitude, longitude, sens, altitude):
    '''This function returns an astropy Earthlocation object based on ROS location values'''
    if sens == 'E':
        location = astropy.coordinates.EarthLocation(longitude*u.deg, latitude*u.deg, altitude*u.m)
    elif sens == 'W':
        location = astropy.coordinates.EarthLocation(-longitude*u.deg, latitude*u.deg, altitude*u.m)
    return location
    
  
def convert_horizon(horizondef,horizontype):
    '''This converts the string of ROS's horizondef into a suitable table of values'''
    lims = horizondef.split("} {")
    lims[0] = lims[0].replace("{","")
    lims[len(lims)-1] = lims[len(lims)-1].replace("}","")
    lims[:]=[x.split(" ") for x in lims]
    if horizontype == "hadec" :
        limites = Table(names =["dec","limrise","limset"], dtype =("f4","f4","f4"))
        return limites
    elif horizontype == "altaz":
        limites = Table(names =["azim","elev"], dtype =("f4","f4"))
    for item in lims :
        limites.add_row(item)
    limites = limites*u.deg
    return limites

def get_db_info(connection, table, entry, keycolumn, keyvalue):
    '''This utilitarian function retrieves a bit of information from a distant table.
    The connection must first be established with pymysql.connect(), then passed in a 1st argument'''
    error = 0
    query = ""
    query = "SELECT " + entry + " FROM " + table + " WHERE " + keycolumn + "= " + keyvalue + ";"
    mycursor = connection.cursor()
    countrow = mycursor.execute(query)
    if countrow != 1 :
        error =1
        print("error fetching data in database")
        value = ""
    value = mycursor.fetchone()[0]
    return error, value
    
  


def site_field(site):
    '''This is a crude definition of each site's FoV'''
    for case in switch(site):
        if case("'Tarot_Calern'"):
            return 1.86
            break
        if case("'Tarot_Chili'"):
            return 1.86
            break
        if case("'Tarot_Reunion'"):
            return 4.2
            break
        if case("'Zadko_Australia'"):
            return 0.5
            break
def site_number(site):
    '''This is a crude definition of each site's targeted number of fields'''
    for case in switch(site):
        if case("'Tarot_Calern'"):
            return 5
            break
        if case("'Tarot_Chili'"):
            return 5
            break
        if case("'Tarot_Reunion'"):
            return 10
            break
        if case("'Zadko_Australia'"):
            return 5
            break
def site_timings(site):
    '''This is a crude definition of each site's timing parameters
    as well as exposure scheme'''
    for case in switch(site):
        if case("'Tarot_Calern'"):
            settime = 30     
            readoutTime = 10
            exptime = 120
            exps = [exptime,exptime,0,0,0,0]
            fil = "1"
            filters = [fil,fil,fil,fil,fil,fil]
            return settime,readoutTime,exps,filters
            break
        if case("'Tarot_Chili'"):
            settime = 30     
            readoutTime = 10
            exptime = 120
            exps = [exptime,exptime,0,0,0,0]
            fil = "1"
            filters = [fil,fil,fil,fil,fil,fil]
            return settime,readoutTime,exps,filters
            break
        if case("'Tarot_Reunion'"):
            settime = 30     
            readoutTime = 10
            exptime = 120
            exps = [exptime,exptime,exptime,0,0,0]
            fil = "0"
            filters = [fil,fil,fil,fil,fil,fil]
            return settime,readoutTime,exps,filters
            break
        if case("'Zadko_Australia'"):
            settime = 90     
            readoutTime = 10
            exptime = 120
            exps = [exptime,exptime,0,0,0,0]
            fil = "1"
            filters = [fil,fil,fil,fil,fil,fil]
            return settime,readoutTime,exps,filters
            break


def replica_is_running():
    '''This simple function checks if replica is running on this machine
    It is useless unless replica runs locally'''
    ps_replica = subprocess.check_output(["ps","-edf"])
    if ps_replica.count("replica") > 0:
        return True
    else:
        return False
