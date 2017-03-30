# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:24:13 2017

@author: RLaugier
"""



# Python standard library imports
import tempfile
import shutil
import sys
import glob
# Third-party imports
import gcn
import gcn.handlers
import gcn.notice_types
import requests
import healpy as hp
import numpy as np
from time import sleep
import lxml

import astropy.coordinates
from astropy.time import Time
from datetime import datetime
import astropy.units as u
from astropy.table import Table

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
            
            
            
            

def download_skymap(myurl):
    #subprocess.check_call(['curl', '-O', '--netrc', myurl])
    subprocess.check_call(['wget','--auth-no-challenge',myurl])

def load_skymap(myfile):
    hpx, header = hp.read_map(myfile, h=True, verbose=True)
    
    return hpx, header

def skymap_properties(hpx):
    print ("Number of pixels:"); print (len(hpx))

    nside = hp.npix2nside(len(hpx))
    print ("The lateral resolution (nside) is:"); print (nside)
    
    sky_area = 4 * 180**2 / np.pi
    print("pixel per degree:")
    print(len(hpx) / sky_area)
    

def ratophi(ra):
    myphi = np.deg2rad(ra)
    myphi = checkphi(myphi)
    return myphi
def phitora(phi):
    ra = np.rad2deg(phi)
    return ra
        
    
def dectotheta(dec):
    theta = 0.5*np.pi - np.deg2rad(dec)
    theta = checktheta(theta)
    return theta
def thetatodec(theta):
    dec = np.rad2deg(0.5*np.pi - theta)
    if dec > 180 : print("error"); return -1000;
    if dec < -180 : print("error"); return -1000;
    else : return dec
    
    
def checkphi(phi):
    phi = phi % (np.pi*2)
    return phi    
    
    
def checktheta(theta):
    theta = theta % np.pi
    return theta

def checkra(ra):
    if type(ra) == float : ra = ra % 360
    else : print("error: need float type for checkra"); return;
    return ra

def getcorners(ra,dec,field):
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


def slicesky(d0, s0,s1,field):
#d0, s0, s1 are variables that allow shifted and skew slicing
#d0<field
#s0<field
#s1<s0
    startpoint = (0,0)
#create a declination slicing
    decgrid = list(np.arange(-90 + d0,90,field))
    mylistra, mylistdec = [], []
    n = 0
#for each declination, create a ra slicing
    for dec in decgrid :
        rastep = field/(np.cos(np.deg2rad(dec)))
        ragrid = list(np.arange(s0 + n*s1, s0 + n*s1 + 360, rastep))
        decfields = []
        for myfield in ragrid:
            decfields.append(dec)
        mylistra.extend(ragrid)
        mylistdec.extend(decfields)
        n += 1


    if len(mylistra) != len(mylistdec) : print ("error, couldn't match ra to dec"); return 1;
    return mylistra, mylistdec
    
    
def get_field_value(hpx,ra,dec,field):
    sq2 = 2**0.5
    nside = hp.npix2nside(len(hpx))
#    pixel = hp.ang2pix(mynside, dectotheta(dec), ratophi(ra))
    xyz = hp.ang2vec(dectotheta(dec),ratophi(ra))
    ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(field /2))#* (sq2+1)/4)here radius seems to be a diameter instead * (sq2+1)/4)
    totdisc = hpx[ipix_disc].sum()
    return totdisc
def get_fast_field_value(hpx,ra,dec,field):
    nside = hp.npix2nside(len(hpx))
    mypix = hp.ang2pix(nside,dectotheta(dec),ratophi(ra))
    return hpx[mypix]
    
def get_fields_value(hpx,myfieldsra,myfieldsdec, field):
    sq2 = 2**0.5
    nside = hp.npix2nside(len(hpx))
    prob=[]
    for indec in range(0, len(myfieldsra)):
#        cornerra, cornerdec = getcorners(keptfields["ra"][indec],keptfields["dec"][indec],field=4.2)
#        xyz = hp.ang2vec(checktheta(dectotheta(cornerdec)),checkphi(ratophi(cornerdec)))
#        hp.query_polygon(nside,xyz)
        xyz = hp.ang2vec(dectotheta(myfieldsdec[indec]),ratophi(myfieldsra[indec]))
        ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(field  /2))#* (sq2+1)/4)here radius seems to be a diameter instead * (sq2+1)/4)
        totdisc = hpx[ipix_disc].sum()
        prob.append(totdisc)
    return prob
    

def calculate_efficiency(hpx, d0, s0, s1, nfields, fieldop):
    nside = hp.npix2nside(len(hpx))
    sq2 = 2**0.5
    keptfields = build_fields(hpx, d0,s0,s1,nfields,fieldop)
    totaldots = np.sum(keptfields["prob"])
    total= 0
    prob_integral = []
    for indec in range(0, len(keptfields)):
#        cornerra, cornerdec = getcorners(keptfields["ra"][indec],keptfields["dec"][indec],field=4.2)
#        xyz = hp.ang2vec(checktheta(dectotheta(cornerdec)),checkphi(ratophi(cornerdec)))
#        hp.query_polygon(nside,xyz)
        xyz = hp.ang2vec(dectotheta(keptfields["dec"][indec]),ratophi(keptfields["ra"][indec]))
        ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(fieldop))#here radius seems to be a diameter instead * (sq2+1)/4)
        totdisc = hpx[ipix_disc].sum()
        prob_integral.append(totdisc)
        total += totdisc
    
    efficiency = total / nfields
    #print (total)
    return total, prob_integral
        
        
def optimize_quin(hpx, nfields, fieldaz):
    mymax, d0max, s0max, s1max = 0,0,0,0
    numberofiterations = 0
    for d0 in np.arange(0,fieldaz*3/4,fieldaz/4):
        for s0 in np.arange(0,fieldaz*3/4,fieldaz/4):
            for s1 in np.arange(-fieldaz/3,fieldaz/3,fieldaz/9):
                value, prob_integral = calculate_efficiency(hpx, d0, s0, s1, nfields, fieldaz)
                numberofiterations += 1
                if value > mymax :
                    mymax, d0max, s0max, s1max = value, d0, s0, s1
    print(numberofiterations, mymax)
    return mymax, d0max, s0max, s1max
 
   
'''
total, a,b,c = optimize_quin(hpx, 5, 4.2)
build_fields(hpx, a, b, c, 5, 4.2)

'''
def build_fields(hpx,d0, s0, s1, nfields, fieldsize):
    nside = hp.npix2nside(len(hpx))
    myfieldsra , myfieldsdec = slicesky(d0,s0,s1,fieldsize)
    
    #prob = get_fields_value(hpx, myfieldsra, myfieldsdec,fieldsize)
    prob = get_fast_field_value(hpx, myfieldsra, myfieldsdec,fieldsize)
    fieldtable = Table([myfieldsra,myfieldsdec,prob], names = ("ra","dec","prob"))
    
    fieldtable.sort(keys = "prob")
    fieldtable.reverse()
    keptfields = fieldtable[0:nfields*5]
    
    total = 0
    prob_integral = []
    for indec in range(0, len(keptfields)):
#        cornerra, cornerdec = getcorners(keptfields["ra"][indec],keptfields["dec"][indec],field=4.2)
#        xyz = hp.ang2vec(checktheta(dectotheta(cornerdec)),checkphi(ratophi(cornerdec)))
#        hp.query_polygon(nside,xyz)
        xyz = hp.ang2vec(dectotheta(keptfields["dec"][indec]),ratophi(keptfields["ra"][indec]))
        ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(fieldsize))#here radius seems to be a diameter instead * (sq2+1)/4)
        totdisc = hpx[ipix_disc].sum()
        prob_integral.append(totdisc)
        total += totdisc
    
    keptfields["proba"] = prob_integral
    keptfields.sort(keys = "proba")
    keptfields.reverse()
    keptfields = keptfields[0:nfields]
    return keptfields


def get_skymap(root):
    """
    Look up URL of sky map in VOEvent XML document,
    download sky map, and parse FITS file.
    """
    # Read out URL of sky map.
    # This will be something like
    # https://gracedb.ligo.org/apibasic/events/M131141/files/bayestar.fits.gz
    skymap_url = root.find(
        "./What/Param[@name='SKYMAP_URL_FITS_BASIC']").attrib['value']

    # Send HTTP request for sky map
    response = requests.get(skymap_url, stream=True)

    # Uncomment to save VOEvent payload to file
    # open('example.xml', 'w').write(payload)

    # Raise an exception unless the download succeeded (HTTP 200 OK)
    response.raise_for_status()

    # Create a temporary file to store the downloaded FITS file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # Save the FITS file to the temporary file
        shutil.copyfileobj(response.raw, tmpfile)
        tmpfile.flush()

        # Uncomment to save FITS payload to file
        # shutil.copyfileobj(reponse.raw, open('example.fits.gz', 'wb'))

        # Read HEALPix data from the temporary file
        skymap, header = hp.read_map(tmpfile.name, h=True, verbose=False)
        header = dict(header)

    # Done!
    return skymap, header


# Function to call every time a GCN is received.
# Run only for notices of type LVC_INITIAL or LVC_UPDATE.
@gcn.handlers.include_notice_types(
    gcn.notice_types.LVC_INITIAL,
    gcn.notice_types.LVC_UPDATE)
def process_gcn(payload, root):
    # Print the alert
    print('Got VOEvent:')
    print(payload)

    # Respond only to 'test' events.
    # VERY IMPORTANT! Replce with the following line of code
    # to respond to only real 'observation' events.
    # if root.attrib['role'] != 'observation': return
    if root.attrib['role'] != 'test': return

    # Respond only to 'CBC' events. Change 'CBC' to "Burst' to respond to only
    # unmodeled burst events.
    if root.find("./What/Param[@name='Group']").attrib['value'] != 'CBC': return

    # Read out integer notice type (note: not doing anythin with this right now)
    notice_type = int(root.find("./What/Param[@name='Packet_Type']").attrib['value'])

    # Read sky map
    skymap, header = get_skymap(root)

'''conn = pymysql.connect(host='tarot9.oca.eu', user='tarot', password=pwd, db='ros')'''
'''idtelescope, latitude, longitude, sens, altitude, horizondef = get_obs_info("'Tarot_Calern'")'''
'''latitude, longitude, altitude, horizondef, horizontype, a = get_obs_info("'Zadko_Australia'")'''

def get_db_info(connection, table, entry, keycolumn, keyvalue):
    error = 0
    query = ""
    query = "SELECT " + entry + " FROM " + table + " WHERE " + keycolumn + "= " + keyvalue + ";"
    mycursor = connection.cursor()
    countrow = mycursor.execute(query)
    if countrow != 1 : error =1; print("error fetching data in database")
    return error, mycursor.fetchone()[0]
    
'''This assumes that the horizondef absissa is monotonous and increasing!!!
This assumes that the intervals cover ALL the possibilities
It should check that the target is inside the interval '''
def interpolate(dataset, target,horizontype):
    print (target)
    if horizontype == "altaz":

        value = 90*u.deg #this default value will prevent observation in case of a problem
        if ((target < dataset["azim"][0])or(target > dataset["azim"][len(dataset["azim"])-1])):
            print ("error, target out of range")
            return value
        for index in np.arange(0,len(dataset["azim"])-1,1):# "-1-1" allows us to go from 0 to the penultimate
            if ((dataset["azim"][index] <= target) and (dataset["azim"][index+1]>= target)):
                #Interpolation of target in
                value = dataset["elev"][index] + (target - dataset["azim"][index]) * (dataset["elev"][index+1] - dataset["elev"][index]) / (dataset["azim"][index+1] - dataset["azim"][index])
                break
        if value == 90*u.deg :
            print ("error: Couldn't interpolate horizon. Azimuth, result = ")
            print(target,value)
            return value
        else :
            print ("horizon elevation at site is")
            print (value)
            return value
    elif horizontype == "hadec":
        print ("this function doesn't support hadec horizon yet")
    else : print ("error: unknown horizon type")

    return value
def checkhorizon(horizondef,horizontype):
    return
def transform_location(latitude, longitude, sens, altitude):
    if sens == 'E':
        location = astropy.coordinates.EarthLocation(longitude*u.deg, latitude*u.deg, altitude*u.m)
    elif sens == 'W':
        location = astropy.coordinates.EarthLocation(-longitude*u.deg, latitude*u.deg, altitude*u.m)
    return location
def checksun(coordinates, time, mylocation):
    observable = 1
    astropy.coordinates.solar_system_ephemeris.set('builtin')
    SunObject = astropy.coordinates.get_sun(time)
    sunaltaz = SunObject.transform_to(astropy.coordinates.AltAz(obstime=time, location=mylocation))
    if sunaltaz.alt >= -5*u.deg:
        observable = 0
    if SunObject.separation(coordinates) <= 30*u.deg:
        observable = 0
        print("Too close to the sun")
    return observable
def checkmoon(coordinates, time, mylocation):
    observable = 1
    astropy.coordinates.solar_system_ephemeris.set('builtin')
    MoonObject = astropy.coordinates.get_moon(time)
    if MoonObject.separation(coordinates) <= 30*u.deg:
        observable = 0
        print("Object too close to the moon")
    return observable
def checkelev(coordinates, time, mylocation, horizondef, horizontype):
    observable = 1
    #objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(time), location=mylocation)
    objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(location=mylocation, obstime=time))
    if objaltaz.alt <= 10*u.deg:
        observable = 0
        print("Object too low (10°)")
    local_horizon = interpolate(horizondef, objaltaz.az, horizontype)
    if objaltaz.alt - local_horizon <= 0*u.deg:
        observable = 0
        print("Object below site horizon")
    return observable
def checkhadec(coordinates, time, mylocation, hadeclims):
    #hadeclims = (limdecmin,limdecmax,limharise,limhaset)
    observable = 1
    #objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(time), location=mylocation)
    loctime = Time(time,location = mylocation)
    LST = loctime.sidereal_time("apparent")
    
    LHA = coordinates.ra + LST
    print ("Hour angle = ", LHA)
    #testing for a valid hour angle
    if LHA >= 24 * u.hourangle or LHA < 0 * u.hourangle :
        print ("hour angle overflow")
        observable = 0
    #Testing if object is in hadec blind spot
    if LHA >= hadeclims[2]*u.deg and LHA <= hadeclims[3]*u.deg:
        observable = 0
        print("Object HA below limits", LHA)
        
    if coordinates.dec <= hadeclims[0]*u.deg:
        observable = 0
        print("Object DEC below limits", LHA)
    if coordinates.dec >= hadeclims[1]*u.deg:
        observable = 0
        print("Object DEC over limits", LHA)
    
    return observable



def site_field(site):
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

'''"https://gracedb.ligo.org/apibasic/events/M131141/files/bayestar.fits.gz"'''
'''"https://gracedb.ligo.org/apibasic/events/G277583/files/skyprobcc_cWB.fits"'''
def main(url,pwd):
    current_time = Time(datetime.utcnow(), scale='utc')
    print("Date is");print(current_time.value)
    print("retrieving skymap from %s"% url); sys.stdout.flush()
    download_skymap(url)
    names = url.split("/")
    names.reverse()
    name = names[0]
    print("loading skymap named %s"% name); sys.stdout.flush()
    hpx, header = load_skymap(name)
##################################################################################
###Site    
    site = "'Tarot_Reunion'"
    
    print ("working on site: %s", site); sys.stdout.flush()
    location, horizondef, horizontype, hadeclims = get_obs_info(site,pwd)
    total, a,b,c = optimize_quin(hpx, site_number(site), site_field(site))
    myfields = build_fields(hpx, a, b, c, site_number(site)*2, site_field(site))
    print (myfields); sys.stdout.flush()
    thefields = clean_table(myfields)
    print (thefields); sys.stdout.flush()
    print ("Observavility from %s, at location %s" % (site, location))
    for index in np.arange(0,len(thefields)-1,1):
        sunok = checksun(thefields["coords"][index], current_time, location)
        moonok = checkmoon(thefields["coords"][index], current_time, location)
        elevok = checkelev(thefields["coords"][index], current_time, location, horizondef,horizontype)
        hadecok = checkhadec(thefields["coords"][index], current_time, location, hadeclims)
        print (index, sunok, moonok, elevok)
    return thefields,location
    
def clean_table(fields):
    coords = astropy.coordinates.SkyCoord(ra=fields["ra"]*u.deg,dec=fields["dec"]*u.deg,frame="fk5")
    fields["coords"] = coords
    del fields["ra"]
    del fields["prob"]
    del fields["dec"]
    return fields
    
def convert_horizon(horizondef,horizontype):
    lims = horizondef.split("} {")
    lims[0] = lims[0].replace("{","")
    lims[len(lims)-1] = lims[len(lims)-1].replace("}","")
    lims[:]=[x.split(" ") for x in lims]
    if horizontype == "hadec" :
        limites = Table(names =["dec","limrise","limset"], dtype =("f4","f4","f4")) 
    elif horizontype == "altaz":
        limites = Table(names =["azim","elev"], dtype =("f4","f4"))
    for item in lims :
        limites.add_row(item)
    limites = limites*u.deg
    return limites

    
def get_obs_info(sitename,pwd):
    conn = pymysql.connect(host='tarot9.oca.eu', user='tarot', password=pwd, db='ros')
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
    return location, myhorizon, horizontype, hadeclims
    