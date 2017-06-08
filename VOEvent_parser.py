# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:24:13 2017

@author: RLaugier
"""



# Python standard library imports
import tempfile
import shutil
import sys
# Third-party imports
import gcn
import gcn.handlers
import gcn.notice_types
import requests
import healpy as hp
import numpy as np

from datetime import datetime
import astropy.coordinates
from astropy.time import Time
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii

import subprocess

from time import sleep

import os


#internal pyros imports
import pyrosutilities
import cadorrest



            
#parsing for global variables
print ("Parsing user password")
try : 
    userpassword = str(sys.argv[1])
except:
    print("error: expected the password for user alert inbetween \"s like \"****\" ")
print ("Parsing database password")
try: 
    databasepassword = str(sys.argv[2])
except:
    print("error: expected the database password for user ros inbetween \"s like \"****\" ")
print ("initiating test counter")
testcounter = 0
testcycle = 100
try:
    runmode = str(sys.argv[3])
    print("mode",runmode)
except:
    runmode = "run"
            
#Defining the functions for the core

def download_skymap(myurl,alertname,name):
    '''This function actually downloads the file from an url then moves it to
    the directory of the corresponding name.'''
    #subprocess.check_call(['curl', '-O', '--netrc', myurl])
    subprocess.check_call(['wget','--auth-no-challenge', myurl])
    shutil.move(name,os.path.join(alertname,name))

def load_skymap(myfile):
    '''Loads a healpix skymap'''
    hpx, header = hp.read_map(myfile, h=True, verbose=True)
    
    return hpx, header

def skymap_properties(hpx):
    '''Reads and displays some of the main properties of a skymap'''
    print ("Number of pixels:"); print (len(hpx))

    nside = hp.npix2nside(len(hpx))
    print ("The lateral resolution (nside) is:"); print (nside)
    
    sky_area = 4 * 180**2 / np.pi
    print("pixel per degree:")
    print(len(hpx) / sky_area)
    

def ratophi(ra):
    '''Converts a RA value into a phi angle for healpy to use. Also uses checkphi
    to check for overflows in the operation
    ra*u.deg -> phi*u.rad'''
    myphi = np.deg2rad(ra)
    myphi = checkphi(myphi)
    return myphi
def phitora(phi):
    '''Converts a phi angle from healpy coordinates into RA angle
    phi*u.rad -> ra*u.deg'''
    ra = np.rad2deg(phi)
    return ra
        
    
def dectotheta(dec):
    '''Converts a DEC value into a theta angle for healpy to use. Also uses checktheta
    to check for overflows in the operation
    dec*u.deg -> theta*u.rad'''
    theta = 0.5*np.pi - np.deg2rad(dec)
    theta = checktheta(theta)
    return theta
def thetatodec(theta):
    '''Converts a theta angle from healpy into a DEC angle
    theta*u.rad -> dec*u.deg'''
    dec = np.rad2deg(0.5*np.pi - theta)
    if dec > 180 : print("error"); return -1000;
    if dec < -180 : print("error"); return -1000;
    else : return dec
    
    
def checkphi(phi):
    '''Checks for a possible overflow in the value of a phi angle
    so that it remains consistant with the system used in healpy'''
    phi = phi % (np.pi*2)
    return phi    
    
    
def checktheta(theta):
    '''Checks for a possible overflow in the value of a phi angle
    so that it remains consistant with the system used in healpy'''
    theta = theta % np.pi
    return theta

def checkra(ra):
    '''Checks for a possible overflow in the value of a RA angle
    so that it remains consistant with the system used in healpy'''
    if type(ra) == float : ra = ra % 360
    else : print("error: need float type for checkra"); return;
    return ra



def slicesky(d0, s0,s1,field):
    '''This function builds a list of coordinates that cover the whole
    sky with FoV of a givent instrument. Parameters d0, s0 and s1 are 
    used to give tiling variations'''
#d0, s0, s1 are variables that allow shifted and skew slicing
#d0<field
#s0<field
#s1<s0

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
    '''This function returns the integral probability inside a disc field of view.
    It is more reliable than the get_fast_field_value, but also slower.'''
    nside = hp.npix2nside(len(hpx))
#    pixel = hp.ang2pix(mynside, dectotheta(dec), ratophi(ra))
    xyz = hp.ang2vec(dectotheta(dec),ratophi(ra))
    ipix_disc = hp.query_disc(nside, xyz, np.deg2rad(field /2))#* (sq2+1)/4)here radius seems to be a diameter instead * (sq2+1)/4)
    totdisc = hpx[ipix_disc].sum()
    return totdisc
def get_fast_field_value(hpx,ra,dec,field):
    '''This returns the skymap pixel value of the center of a field'''
    nside = hp.npix2nside(len(hpx))
    mypix = hp.ang2pix(nside,dectotheta(dec),ratophi(ra))
    return hpx[mypix]
    
def get_fields_value(hpx,myfieldsra,myfieldsdec, field):
    '''This returns the disc integral values for a whole list of fields'''
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
    '''This function determines the score of a given set of tiling parameters for the optimization.
    The figure of merit is the sum value of the for the <nfields> higher fields of the tiling'''
    nside = hp.npix2nside(len(hpx))

    keptfields = build_fields(hpx, d0,s0,s1,nfields,fieldop)
#    totaldots = np.sum(keptfields["prob"])
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
    
#    efficiency = total / nfields
    #print (total)
    return total, prob_integral
        
        
def optimize_quin(hpx, nfields, fieldaz):
    '''This function returns uptimized tiling parameters based on their calculate_efficiency() score for <nfields>'''
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
    '''This function is designed to build a list of the <nfield> fields that have 
    the highest disc integral value on the hpx map.
    Usually, this will be used with d0,s0 and s1 previously selected by optimize_quin()'''
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
    gcn.notice_types.LVC_TEST)
def process_gcn(payload, root):
    global testcounter
    
    # Print the alert
    print('Got VOEvent:')
    #print(payload)

    # Respond only to 'test' events.
    # VERY IMPORTANT! Replce with the following line of code
    # to respond to only real 'observation' events.
    # if root.attrib['role'] != 'observation': return
    
    #This is the test sequence procedure, we run global in test mode once every x time (defined by testcycle)
    if root.attrib['role'] == 'test' :
        print("This is a test: checking testcounter"); sys.stdout.flush()
        print(testcounter); sys.stdout.flush()
        if testcounter == 0:
            print("We run the test this time")
            print(payload)
            skymap_url = root.find("./What/Param[@name='SKYMAP_URL_FITS_BASIC']").attrib['value']
            process_global(skymap_url,userpassword,databasepassword,test=True)
        else :
            print ("The test counter is at %i: not doing the drill this time.")
        testcounter += 1
        if testcounter >= testcycle:
            testcounter=0
    
    #This is what happens if the alert is NOT A TEST: we run global in non-test mode
    if root.attrib['role'] == 'observation' :
        print(payload)
        skymap_url = root.find("./What/Param[@name='SKYMAP_URL_FITS_BASIC']").attrib['value']
        process_global(skymap_url,userpassword,databasepassword,test=False)


    # Respond only to 'CBC' events. Change 'CBC' to "Burst' to respond to only
    # unmodeled burst events.
    if root.find("./What/Param[@name='Group']").attrib['value'] != 'CBC': return

    # Read out integer notice type (note: not doing anythin with this right now)
    notice_type = int(root.find("./What/Param[@name='Packet_Type']").attrib['value'])

    # Read sky map
    skymap, header = get_skymap(root)




def interpolate(dataset, target,horizontype):
    '''This function is used to interpolate an horizon definition.
    It works with altaz horizons but was adapted to ROS hadec horizons (pourly standardized)
    This assumes that the horizondef absissa is monotonous and increasing!!!
    This assumes that the intervals cover ALL the possibilities
    It should check that the target is inside the interval '''
    #print (target)
    if horizontype == "altaz":

        value = 90*u.deg #this default value will prevent observation in case of a problem
        if ((target < dataset["azim"][0])or(target > dataset["azim"][len(dataset["azim"])-1])):
            print ("error, target out of range")
            return value
        for index in np.arange(0,len(dataset["azim"])-1,1):# "-1" allows us to go from 0 to the penultimate
            #print("Between",dataset["azim"][index],"and ",dataset["azim"][index+1])
            if ((dataset["azim"][index] <= target) and (dataset["azim"][index+1]>= target)):
                #Interpolation of target in
                value = dataset["elev"][index] + (target - dataset["azim"][index]) * (dataset["elev"][index+1] - dataset["elev"][index]) / (dataset["azim"][index+1] - dataset["azim"][index])
                break
        if value == 90*u.deg :
            print ("error: Couldn't interpolate horizon. Azimuth, result =============================== ")
            print(target,value)
            return value
        else :
            #print ("horizon elevation at site is")
            #print (value)
            return value
    elif horizontype == "hadec":
        print ("this function doesn't support hadec horizon yet")
    else : print ("error: unknown horizon type")

    return value
def checkhorizon(horizondef,horizontype):
    return

def checksun(coordinates, time, mylocation):
    '''This function checks for the presence of night at the site(-5° of sun elevation)
    and an agular separation to the sun greater than 30°'''
    observable = 1
    astropy.coordinates.solar_system_ephemeris.set('builtin')
    SunObject = astropy.coordinates.get_sun(time)
    sunaltaz = SunObject.transform_to(astropy.coordinates.AltAz(obstime=time, location=mylocation))
    if sunaltaz.alt >= -5*u.deg:
        observable = 0
        return observable
    if SunObject.separation(coordinates) <= 30*u.deg:
        observable = 0
        print("Too close to the sun")
    return observable
def checkmoon(coordinates, time, mylocation):
    '''This checks for a moon distance larger than 30°'''
    observable = 1
    astropy.coordinates.solar_system_ephemeris.set('builtin')
    MoonObject = astropy.coordinates.get_moon(time)
    if MoonObject.separation(coordinates) <= 30*u.deg:
        observable = 0
        print("Object too close to the moon")
    return observable
def checkelev(coordinates, time, mylocation, horizondef, horizontype):
    '''This makes a basic check for an elevation higher tha0 10° AND in case 
    of an altaz horizon definition, checks for an elevation higher than this horizon.'''
    observable = 1
    #objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(time), location=mylocation)
    objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(location=mylocation, obstime=time))
    if objaltaz.alt <= 10*u.deg:
        observable = 0
        #print("Object too low (10°)")
        return observable
    #This condition is assumed sufficient for unsupported hadec horizons
    if horizontype == "hadec":
        return observable
    local_horizon = interpolate(horizondef, objaltaz.az, horizontype)
    if objaltaz.alt - local_horizon <= 0*u.deg:
        observable = 0
        print("Object below site horizon")
    return observable
def checkhadec(coordinates, time, mylocation, hadeclims):
    '''This function checks for compliance to simple hadec mount limits, in 
    a format compliant to the ROS telescopes database table'''
    #hadeclims = (limdecmin,limdecmax,limharise,limhaset)
    observable = 1
    #objaltaz = coordinates.transform_to(astropy.coordinates.AltAz(time), location=mylocation)
    loctime = Time(time,location = mylocation)
    LST = loctime.sidereal_time("apparent")
    
    LHA = coordinates.ra + LST
    #testing for a valid hour angle
    if LHA >= 24 * u.hourangle or LHA < 0 * u.hourangle :
        #print ("hour angle overflow",LHA)
        LHA = LHA - 24*u.hourangle
        #print ("corrected:",LHA)
    #Testing if object is in hadec blind spot
    if LHA >= hadeclims[2]*u.deg and LHA <= hadeclims[3]*u.deg:
        observable = 0
        print("Object HA below limits", LHA)
        return observable
        
    if coordinates.dec <= hadeclims[0]*u.deg:
        observable = 0
        print("Object DEC below limits=================================", coordinates.dec)
        return observable
    if coordinates.dec >= hadeclims[1]*u.deg:
        observable = 0
        print("Object DEC over limits==================================", coordinates.dec)
        return observable
    
    return observable
def check_declination(latitude,declination):
    '''This method is a preliminary check designed to eliminate targets never observable from
    the latitude of the observatory, and reduce calculation steps. This assumes an elevation margin of 10°'''
    observable = 1
    if latitude > 0:
        if declination <  (latitude - 80*u.deg):
            observable = 0
            print("field rejected for declination too low",declination)
    elif latitude < 0:
        if declination > latitude + 80*u.deg:
            observable = 0
            print("field rejected for declination too low",declination)
    return observable

def process_global(url,pwd,dbpwd,test=False):
    '''This function is designed to do all the steps required for the scheduling of
    a whole GW event followup for all the telescopes in the network, based solely on the url of a skymap.
    warning: the URL is used assume the id of the event, so let's hope they do not change their url scheme for now
    In case the alert is a test, the scenes will be teleted after 10s'''
    if test == True:
        print ("Ok, let's go through the drill one more time")
    else :
        print ("Ok this is for real now! This is not an exercise!")
    start_time = Time(datetime.utcnow(), scale='utc')
    sitenames = ["'Tarot_Calern'","'Tarot_Chili'","'Tarot_Reunion'"]
    siteids = [1,2,8]
    scenes = Table()
    name = os.path.basename(url)
    alertname = url.split("/")[5]
    if not os.path.isdir(alertname):
        os.makedirs(alertname)
    print("retrieving skymap from %s"% url); sys.stdout.flush()
    download_skymap(url, alertname, name)

    
    print("loading skymap named %s"% name); sys.stdout.flush()
    hpx, header = load_skymap(os.path.join(alertname,name))
    for site in sitenames:
        sitescenes = Table()
        fields,mylocation,sitescenes = main(hpx, header, site, pwd,dbpwd)
        #this fixes problems arising when a table was empty
        if len(sitescenes)>0 and type(sitescenes) == astropy.table.table.Table:
            if len(scenes)>0:
                print ("joining")
                scenes.pprint(max_width=-1)
                sitescenes.pprint(max_width=-1)
                scenes = astropy.table.vstack([scenes,sitescenes])
            else :
                scenes = sitescenes
    print (scenes)
    
    

    
    settime,readoutTime,exps,filters = pyrosutilities.site_timings(site)

    idreq = cadorrest.post_request(alertname + "autogen","0","90",pwd)
    idscene = []
    ra = []
    dec = []
    timeisot = []
    for index in np.arange(0,len(scenes),1):
        #cadorrest.post_scene(prefix,idreq,entry,exps,filters,pwd):
        myidscene, myra, mydec, mytimeisot = cadorrest.post_scene(alertname+"_",idreq,scenes[index],exps,filters,pwd)
        idscene.append(myidscene)
        ra.append(myra)
        dec.append(mydec)
        timeisot.append(mytimeisot)
    scenes ["idscene"] = idscene
    scenes ["ra"] = ra
    scenes ["dec"] = dec
    scenes ["timeisot"] = timeisot
        
    end_time = Time(datetime.utcnow(), scale='utc')
    length = end_time - start_time
    print ("Executed in ",length.sec,"seconds")
    print ("Waiting for planification (10s)")
    sys.stdout.flush()
    sleep(10)
    
    #Downloading planification logs
    print("Keeping a little souvenir"); sys.stdout.flush()
    for idteles in siteids:
        planiurl = "http://cador.obs-hp.fr/ros/sequenced" + str(idteles) + ".txt"
        rejecurl = "http://cador.obs-hp.fr/ros/rejected" + str(idteles) + ".txt"
        try :
            download_skymap(planiurl,alertname,"sequenced"+str(idteles)+".txt")
            download_skymap(rejecurl,alertname,"rejected"+str(idteles)+".txt")
            print ("Succesfully downloaded plani files")
        except:
            print ("error copying plani")
    sys.stdout.flush()    
    ascii.write(scenes, os.path.join(alertname,'Planification_table.csv'), format='csv', fast_writer=False)
    if test == True:
        print ("This was just an exercise, let's delete the request now")
        cadorrest.remove_request(idreq,pwd)
    else :
        print ("This was for real: the scenes will be observed")
#        for i in np.arange(0,900,10):
#            if pyrosutilities.replica_is_running():
#                print("Waiting until replica has finished")
#                sleep (10)
#            else:
#                print("Lauching replica to propagate planification")
#                #subprocess.check_call("nohup", "php", "/srv/develop/ros_private_cador/src/replica2/replica_slow.php", "1>", "/srv/www/htdocs/ros/logs/replica/replica_slow.txt")
#                try:
#                    subprocess.check_call("php", "/srv/develop/ros_private_cador/src/replica2/replica_slow.php")
#                    print("Replica finished successfuly, we are done, here")
#                except:
#                    print("Replica was probably crashed: waited for %i and got no response"% i)
#                break
        print("Unfortunately, we cannot run replica from vega yet. We have to be patient and wait for it to run on schedule")
        

    
    
    return scenes
        

'''"https://gracedb.ligo.org/apibasic/events/M131141/files/bayestar.fits.gz"'''
'''"https://gracedb.ligo.org/apibasic/events/G277583/files/skyprobcc_cWB.fits"'''
def main(hpx,header,site,pwd,dbpwd):
    '''This function handles the scenes and planning creation for a single telescope.
    It returns a list of scene entries in a table, ready to be submited to CADOR'''
    current_time = Time(datetime.utcnow(), scale='utc')
    print("Date is");print(current_time.value)

##################################################################################
###Site    
    #site = "'Tarot_Reunion'"
    nfields = pyrosutilities.site_number(site)
    print ("working on site: %s"% (site)); sys.stdout.flush()
    location, horizondef, horizontype, hadeclims, idtelescope = pyrosutilities.get_obs_info(site,dbpwd)
    total, a,b,c = optimize_quin(hpx, nfields, pyrosutilities.site_field(site))
    myfields = build_fields(hpx, a, b, c, nfields*2, pyrosutilities.site_field(site))
    print (myfields); sys.stdout.flush()
    thefields = clean_table(myfields)
    print (thefields); sys.stdout.flush()
    print(thefields["coords"][0])
    print ("Observavility from %s, at location %s" % (site, location))
    toremove = []
    for index in np.arange(0,len(thefields),1):
        if check_declination(location.latitude,thefields[index]["coords"].dec) == 0:
            toremove.append(index)
    thefields.remove_rows(toremove)
    mycyclegrid,scenelength = build_cyclegrid(15,site,24)
    #Determining observability to remove excess fields
    fieldobservability = []
    for i in np.arange(0,len(thefields),1):
        obsevability = []
        for j in np.arange(0,len(mycyclegrid),1):
            result = is_observable(thefields["coords"][i],current_time + mycyclegrid["date"][j]*u.second,location,horizondef,horizontype,hadeclims)
            obsevability.append(result)
        print (obsevability)
        fieldobservability.append(obsevability)
    #thefields["obsevability"]=fieldobservability
    thefields["observability"] = fieldobservability
    print (thefields,fieldobservability)
    toremove=[]
    for i in np.arange(0,len(thefields),1):
        if np.count_nonzero(thefields["observability"][i])<3:
            toremove.append(i)
    thefields.remove_rows(toremove)
    print(thefields)
    if len(thefields)>nfields:
        thefields=thefields[0:nfields]
    elif len(thefields)==0:
        return thefields,location,fieldobservability
    thefields["index"]=np.arange(0,len(thefields),1)
    print(thefields)
    print("proba covered:",np.sum(thefields["proba"]))
    #unable to sort the fields at this point because of SkyCoord object.
    #thefields["temp"]=thefields["coords"].ra
    mycyclegrid,scenelength = build_cyclegrid(len(thefields),site,48)
    
    scenes = []
    for i in np.arange(0,len(thefields),1):
        obsevability = []
        timeindex = 0
        for j in np.arange(0,len(mycyclegrid),1):
            exacttime = mycyclegrid["date"][j]*u.second + i*scenelength*u.second + current_time
            result = is_observable(thefields["coords"][i],exacttime,location,horizondef,horizontype,hadeclims)
            if result == 1:
                scenes.append([site,thefields["index"][i],timeindex,exacttime,thefields["coords"][i]])
                timeindex += 1
        
        fieldobservability.append(obsevability)
    revscenes = np.transpose(scenes)
    thescenes = Table()
    thescenes["site"]=revscenes[0]
    thescenes["index"]=revscenes[1]
    thescenes["tindex"]=revscenes[2]
    thescenes["time"]=revscenes[3]
    thescenes["coords"]=revscenes[4]
    thescenes["idtelescope"] = idtelescope

    return thefields,location,thescenes
    
def is_observable(coords,time,location,horizondef,horizontype,hadeclims):
    '''Thid function regroups all observability check into one'''
    sunok = checksun(coords, time, location)
    #Here the sunok is checked and return first to save computation time
    #since night/day condition is a major cause of non-observability
    if sunok == 0: return 0
    moonok = checkmoon(coords, time, location)
    elevok = checkelev(coords, time, location, horizondef,horizontype)
    hadecok = checkhadec(coords, time, location, hadeclims)
    #print (sunok, moonok, elevok,hadecok)
    observable = sunok * moonok * elevok * hadecok
    return observable
    

def next_sunset(mylocation, mytime):
    '''This function returns a rough estimation of the next sunset time'''
    astropy.coordinates.solar_system_ephemeris.set('builtin')
    SunObject = astropy.coordinates.get_sun(mytime)
    sunaltaz = SunObject.transform_to(astropy.coordinates.AltAz(obstime=mytime, location=mylocation))
    if sunaltaz.alt > -10*u.deg :
        night = 0
    else :
        night = 1
    set_time=-100*u.hour
    for i in np.arange(0, 24, 0.25):
        if night == 0:
            
            time = mytime + i*u.hour
            sunaltaz = SunObject.transform_to(astropy.coordinates.AltAz(obstime=time, location=mylocation))
            if night == 0 and sunaltaz.alt < -10*u.deg:
                set_time = time
                break
        elif night == 1:
            night = 1
    return set_time
def build_cyclegrid(number,site,period):
    '''This function returns a raw time grid for a certain that allow the site's telescope to observe
    the entire <number> of fields in a single cycle. The grid is calculated over a given period (hours)'''
    settime,readoutTime,exps,filters = pyrosutilities.site_timings(site)
    timegrid = []
    
    #scenelength = sum(images[np.nonzero(images)]) + len(images[np.nonzero(images)]) * settime
    totalexpscene = np.sum(exps)
    scenelength = totalexpscene + np.count_nonzero(exps) * readoutTime 
    timeshift = scenelength + settime
    freetime = 0
    cycleTime = number * timeshift + freetime
    length = 60 * 60 * period
    for i in np.arange(0,length,cycleTime):
        thetime = i     
        timegrid.append(thetime)
    timetable = Table()
    timetable["date"]=timegrid
    return timetable,scenelength
def build_finegrid(number,site):
    '''This is similar to build_cyclegrid() but it takes into account the time shift
    in the scheduling of one field to the next.'''
    settime,readoutTime,exps,filters = pyrosutilities.site_timings(site)
    timegrid = []
    
    #scenelength = sum(images[np.nonzero(images)]) + len(images[np.nonzero(images)]) * settime
    totalexpscene = np.sum(exps)
    scenelength = totalexpscene + np.count_nonzero(exps) * readoutTime 
    timeshift = scenelength + settime
    freetime = 0
    cycleTime = number * timeshift + freetime
    length = 60 * 60 * 24
    for i in np.arange(0,length,cycleTime):
        thetime = Time(i*u.second,format=u'cxcsec')     
        timegrid.append(thetime)
    timetable = Table()
    timetable["date"]=timegrid
    return timetable
        
    
def clean_table(fields):
    '''This function just translates the rough fields table into a nice
    table of SkyCoords() objects'''
    coords = astropy.coordinates.SkyCoord(ra=fields["ra"]*u.deg,dec=fields["dec"]*u.deg,frame="fk5")
    fields["coords"] = coords
    del fields["ra"]
    del fields["prob"]
    del fields["dec"]
    return fields
    

if runmode == "run":
    print ("the program is now ready: listening for events")
    sys.stdout.flush()
    gcn.listen(port=8096, handler=process_gcn)
elif runmode =="tools":
    print ("The tools are loaded, ready for interactive mode (use ipython)")
