#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on 七月 17, 2026, at 15:27
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = '2Back_nblocks'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'session': '',
    'nBlocks': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 960]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_1_5_1_%s_%s_%s' % (expInfo['participant'], expName, expInfo['session'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\PychopyEx\\2back_num\\2Back_nblocks_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instr_key_resp') is None:
        # initialise instr_key_resp
        instr_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp',
        )
    # create speaker 'instr_sound'
    deviceManager.addDevice(
        deviceName='instr_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_2') is None:
        # initialise instr_key_resp_2
        instr_key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_2',
        )
    # create speaker 'instr_2_sound'
    deviceManager.addDevice(
        deviceName='instr_2_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_3') is None:
        # initialise instr_key_resp_3
        instr_key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_3',
        )
    # create speaker 'instr_3_sound'
    deviceManager.addDevice(
        deviceName='instr_3_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_3_1') is None:
        # initialise instr_key_resp_3_1
        instr_key_resp_3_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_3_1',
        )
    # create speaker 'instr_3_1_sound'
    deviceManager.addDevice(
        deviceName='instr_3_1_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    # create speaker 'instr_3_2_sound'
    deviceManager.addDevice(
        deviceName='instr_3_2_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # create speaker 'instr_3_3_sound'
    deviceManager.addDevice(
        deviceName='instr_3_3_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    # create speaker 'instr_3_4_sound'
    deviceManager.addDevice(
        deviceName='instr_3_4_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    # create speaker 'instr_3_5_sound'
    deviceManager.addDevice(
        deviceName='instr_3_5_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # create speaker 'instr_3_6_sound'
    deviceManager.addDevice(
        deviceName='instr_3_6_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    # create speaker 'instr_3_7_sound'
    deviceManager.addDevice(
        deviceName='instr_3_7_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    # create speaker 'instr_3_8_sound'
    deviceManager.addDevice(
        deviceName='instr_3_8_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    # create speaker 'instr_3_9_sound'
    deviceManager.addDevice(
        deviceName='instr_3_9_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    # create speaker 'instr_3_8_sound_2'
    deviceManager.addDevice(
        deviceName='instr_3_8_sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    # create speaker 'instr_3_9_sound_2'
    deviceManager.addDevice(
        deviceName='instr_3_9_sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    # create speaker 'instr_3_6_sound_2'
    deviceManager.addDevice(
        deviceName='instr_3_6_sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_11') is None:
        # initialise key_resp_11
        key_resp_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_11',
        )
    # create speaker 'instr_3_9_sound_3'
    deviceManager.addDevice(
        deviceName='instr_3_9_sound_3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_into_prac') is None:
        # initialise key_into_prac
        key_into_prac = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_into_prac',
        )
    if deviceManager.getDevice('instr_key_resp_4') is None:
        # initialise instr_key_resp_4
        instr_key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_4',
        )
    # create speaker 'instr_4_sound'
    deviceManager.addDevice(
        deviceName='instr_4_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_8') is None:
        # initialise instr_key_resp_8
        instr_key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_8',
        )
    # create speaker 'instr_4_sound_2'
    deviceManager.addDevice(
        deviceName='instr_4_sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_6') is None:
        # initialise instr_key_resp_6
        instr_key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_6',
        )
    # create speaker 'instr_4_1_sound'
    deviceManager.addDevice(
        deviceName='instr_4_1_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('instr_key_resp_7') is None:
        # initialise instr_key_resp_7
        instr_key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_7',
        )
    # create speaker 'instr_4_2_sound'
    deviceManager.addDevice(
        deviceName='instr_4_2_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('prac_rest_key_resp') is None:
        # initialise prac_rest_key_resp
        prac_rest_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_rest_key_resp',
        )
    # create speaker 'prac_rest_sound'
    deviceManager.addDevice(
        deviceName='prac_rest_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('prac_key_resp') is None:
        # initialise prac_key_resp
        prac_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_key_resp',
        )
    # create speaker 'prac_feedb_sound'
    deviceManager.addDevice(
        deviceName='prac_feedb_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('prac_block_fb_key_resp') is None:
        # initialise prac_block_fb_key_resp
        prac_block_fb_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac_block_fb_key_resp',
        )
    if deviceManager.getDevice('instr_key_resp_5') is None:
        # initialise instr_key_resp_5
        instr_key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key_resp_5',
        )
    # create speaker 'instr_sound_5'
    deviceManager.addDevice(
        deviceName='instr_sound_5',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index='-1',
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('block_fb_key_resp') is None:
        # initialise block_fb_key_resp
        block_fb_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='block_fb_key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instr" ---
    welpage = visual.TextStim(win=win, name='welpage',
        text='欢迎来到2-back实验！\n\n在这个实验中，您会依序在屏幕上看到不同的数字。\n\n如果现在这个数字和往前数第2个数字是一样的，\n请按 z 键，\n\n如果现在这个数字和往前数第2个数字是不同的，\n请按 m 键。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp = keyboard.Keyboard(deviceName='instr_key_resp')
    instr_sound = sound.Sound(
        'audio/instr.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_sound',    name='instr_sound'
    )
    instr_sound.setVolume(1.0)
    # Run 'Begin Experiment' code from code_9
    n_blocks = int(expInfo['nBlocks'])
    
    # --- Initialize components for Routine "instr_2" ---
    welpage_2 = visual.TextStim(win=win, name='welpage_2',
        text='例如，在下面这种情况下，\n\n当前数字和往前数第2个数字是一样的，\n所以应该按 z 键。',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welpage_2_2 = visual.TextStim(win=win, name='welpage_2_2',
        text='请按 z 键继续。',
        font='Heiti SC',
        pos=(0, -0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    same_e_g = visual.ImageStim(
        win=win,
        name='same_e_g', 
        image='sm_example_new.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5,0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    instr_key_resp_2 = keyboard.Keyboard(deviceName='instr_key_resp_2')
    instr_2_sound = sound.Sound(
        'audio/instr_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_2_sound',    name='instr_2_sound'
    )
    instr_2_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3" ---
    welpage_3 = visual.TextStim(win=win, name='welpage_3',
        text='在下面这种情况下，\n\n当前数字和往前数第2个数字是不同的，\n所以应该按 m 键。',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welpage_3_2 = visual.TextStim(win=win, name='welpage_3_2',
        text='请按 m 键继续。',
        font='Heiti SC',
        pos=(0,-0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    different_e_g = visual.ImageStim(
        win=win,
        name='different_e_g', 
        image='diff_example_new.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    instr_key_resp_3 = keyboard.Keyboard(deviceName='instr_key_resp_3')
    instr_3_sound = sound.Sound(
        'audio/instr_3.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_sound',    name='instr_3_sound'
    )
    instr_3_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_1" ---
    welpage_3_1 = visual.TextStim(win=win, name='welpage_3_1',
        text='每次实验开头的前两个数字无法与其它数字进行比对，\n因此无需做出反应，\n\n也就是说，您需要从第三个数字呈现时开始做出反应。\n\n\n请按空格键继续。\n',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_3_1 = keyboard.Keyboard(deviceName='instr_key_resp_3_1')
    instr_3_1_sound = sound.Sound(
        'audio/instr_3_1.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_1_sound',    name='instr_3_1_sound'
    )
    instr_3_1_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_2" ---
    text_45 = visual.TextStim(win=win, name='text_45',
        text='试一试',
        font='Heiti SC',
        pos=(0,0.4), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_9 = visual.TextStim(win=win, name='text_9',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text = visual.TextStim(win=win, name='text',
        text='5',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    instr_3_2_sound = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_2_sound',    name='instr_3_2_sound'
    )
    instr_3_2_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_3" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='正确答案是 不作反应 ，因为往前数两个没有数字。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    instr_3_3_sound = sound.Sound(
        'audio/instr_3_3.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_3_sound',    name='instr_3_3_sound'
    )
    instr_3_3_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_4" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='3\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr_3_4_sound = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_4_sound',    name='instr_3_4_sound'
    )
    instr_3_4_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_5" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='正确答案同样是 不作反应 ，因为往前数两个没有数字。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    instr_3_5_sound = sound.Sound(
        'audio/instr_3_5.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_5_sound',    name='instr_3_5_sound'
    )
    instr_3_5_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_6" ---
    text_12 = visual.TextStim(win=win, name='text_12',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？\n(提示：您需要回忆刚才呈现过的数字，做出判断）',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text='7\n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_11 = visual.TextStim(win=win, name='text_11',
        text='相同                      不同\n 按z键                    按m键',
        font='Heiti SC',
        pos=(0, -0.2), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    instr_3_6_sound = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_6_sound',    name='instr_3_6_sound'
    )
    instr_3_6_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_7" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='正确答案是 不同 ，因为当前数字跟往前数第2个数字不同。\n',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='7\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_13 = visual.TextStim(win=win, name='text_13',
        text='3',
        font='Open Sans',
        pos=(-0.1, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='5',
        font='Open Sans',
        pos=(-0.2, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_14 = visual.TextStim(win=win, name='text_14',
        text='请按空格键继续。',
        font='Heiti SC',
        pos=(0, -0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    instr_3_7_sound = sound.Sound(
        'audio/instr_3_7.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_7_sound',    name='instr_3_7_sound'
    )
    instr_3_7_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_8" ---
    text_15 = visual.TextStim(win=win, name='text_15',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？\n(提示：您需要回忆刚才呈现过的数字，做出判断）',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_16 = visual.TextStim(win=win, name='text_16',
        text='3\n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_17 = visual.TextStim(win=win, name='text_17',
        text='相同                      不同\n 按z键                    按m键',
        font='Heiti SC',
        pos=(0, -0.2), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    instr_3_8_sound = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_8_sound',    name='instr_3_8_sound'
    )
    instr_3_8_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_9" ---
    text_18 = visual.TextStim(win=win, name='text_18',
        text='正确答案是 相同 ，因为当前数字跟往前数第2个数字是一样的。\n',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_19 = visual.TextStim(win=win, name='text_19',
        text='3\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.3569, 1.0000, -0.6314], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_23 = visual.TextStim(win=win, name='text_23',
        text='7',
        font='Open Sans',
        pos=(-0.1, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_20 = visual.TextStim(win=win, name='text_20',
        text='3',
        font='Open Sans',
        pos=(-0.2, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[0.3569, 1.0000, -0.6314], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_21 = visual.TextStim(win=win, name='text_21',
        text='5',
        font='Open Sans',
        pos=(-0.3, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text_22 = visual.TextStim(win=win, name='text_22',
        text='请按空格键继续。',
        font='Heiti SC',
        pos=(0, -0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    instr_3_9_sound = sound.Sound(
        'audio/instr_3_9.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_9_sound',    name='instr_3_9_sound'
    )
    instr_3_9_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_10" ---
    text_24 = visual.TextStim(win=win, name='text_24',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？\n(提示：您需要回忆刚才呈现过的数字，做出判断）',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_25 = visual.TextStim(win=win, name='text_25',
        text='7\n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_26 = visual.TextStim(win=win, name='text_26',
        text='相同                      不同\n 按z键                    按m键',
        font='Heiti SC',
        pos=(0, -0.2), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    instr_3_8_sound_2 = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_8_sound_2',    name='instr_3_8_sound_2'
    )
    instr_3_8_sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_11" ---
    text_27 = visual.TextStim(win=win, name='text_27',
        text='正确答案是 相同 ，因为当前数字跟往前数第2个数字是一样的。\n',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_28 = visual.TextStim(win=win, name='text_28',
        text='7\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.3569, 1.0000, -0.6314], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_29 = visual.TextStim(win=win, name='text_29',
        text='3',
        font='Open Sans',
        pos=(-0.1, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_30 = visual.TextStim(win=win, name='text_30',
        text='7',
        font='Open Sans',
        pos=(-0.2, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[0.3569, 1.0000, -0.6314], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_31 = visual.TextStim(win=win, name='text_31',
        text='3',
        font='Open Sans',
        pos=(-0.3, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text_41 = visual.TextStim(win=win, name='text_41',
        text='5',
        font='Open Sans',
        pos=(-0.4, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text_32 = visual.TextStim(win=win, name='text_32',
        text='请按空格键继续。',
        font='Heiti SC',
        pos=(0, -0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    instr_3_9_sound_2 = sound.Sound(
        'audio/instr_3_9.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_9_sound_2',    name='instr_3_9_sound_2'
    )
    instr_3_9_sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_12" ---
    text_33 = visual.TextStim(win=win, name='text_33',
        text='请判断：这个数字与往前数第2个数字 相同 还是 不同？\n(提示：您需要回忆刚才呈现过的数字，做出判断）',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_34 = visual.TextStim(win=win, name='text_34',
        text='2\n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_35 = visual.TextStim(win=win, name='text_35',
        text='相同                      不同\n 按z键                    按m键',
        font='Heiti SC',
        pos=(0, -0.2), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    instr_3_6_sound_2 = sound.Sound(
        'audio/instr_3_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_6_sound_2',    name='instr_3_6_sound_2'
    )
    instr_3_6_sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_3_13" ---
    text_36 = visual.TextStim(win=win, name='text_36',
        text='正确答案是 不同 ，因为当前数字跟往前数第2个数字不同。\n',
        font='Heiti SC',
        pos=(0, 0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_37 = visual.TextStim(win=win, name='text_37',
        text='2\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_38 = visual.TextStim(win=win, name='text_38',
        text='7',
        font='Open Sans',
        pos=(-0.1, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_39 = visual.TextStim(win=win, name='text_39',
        text='3',
        font='Open Sans',
        pos=(-0.2, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    text_40 = visual.TextStim(win=win, name='text_40',
        text='7',
        font='Open Sans',
        pos=(-0.3, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    text_42 = visual.TextStim(win=win, name='text_42',
        text='3',
        font='Open Sans',
        pos=(-0.4, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    text_43 = visual.TextStim(win=win, name='text_43',
        text='5',
        font='Open Sans',
        pos=(-0.5, 0.03), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    text_44 = visual.TextStim(win=win, name='text_44',
        text='请按空格键继续。',
        font='Heiti SC',
        pos=(0, -0.35), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    instr_3_9_sound_3 = sound.Sound(
        'audio/instr_3_7.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_3_9_sound_3',    name='instr_3_9_sound_3'
    )
    instr_3_9_sound_3.setVolume(1.0)
    
    # --- Initialize components for Routine "end_prac_judge" ---
    into_prac = visual.TextStim(win=win, name='into_prac',
        text='重复实验说明请按“1”\n\n进入练习请按“4”',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_into_prac = keyboard.Keyboard(deviceName='key_into_prac')
    
    # --- Initialize components for Routine "instr_4" ---
    welpage_4 = visual.TextStim(win=win, name='welpage_4',
        text='现在请试一试。\n\n请保持注视屏幕上的“+”注视点，\n并在看到数字后立即做出反应。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_4 = keyboard.Keyboard(deviceName='instr_key_resp_4')
    instr_4_sound = sound.Sound(
        'audio/instr_4.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_4_sound',    name='instr_4_sound'
    )
    instr_4_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_4_3" ---
    welpage_7 = visual.TextStim(win=win, name='welpage_7',
        text='您的任务是判断当前显示的数字是否与\n往前数第2个位置显示的数字相同。\n\n如果相同，请按下键盘上的 z 键；\n如果不同，请按下 m 键。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_8 = keyboard.Keyboard(deviceName='instr_key_resp_8')
    instr_4_sound_2 = sound.Sound(
        'audio/instr_4_3.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_4_sound_2',    name='instr_4_sound_2'
    )
    instr_4_sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_4_1" ---
    welpage_5 = visual.TextStim(win=win, name='welpage_5',
        text='请注意，在下面的实验中，\n数字会很快消失，\n因此，请试着在保证正确的同时尽可能快地回答。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_6 = keyboard.Keyboard(deviceName='instr_key_resp_6')
    instr_4_1_sound = sound.Sound(
        'audio/instr_4_1.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_4_1_sound',    name='instr_4_1_sound'
    )
    instr_4_1_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "instr_4_2" ---
    welpage_6 = visual.TextStim(win=win, name='welpage_6',
        text='如果您回答错误，继续回答下一题即可。\n记得，回答完之后把手指保持在 z 键和 m 键上。\n\n\n请按空格键继续。\n',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_7 = keyboard.Keyboard(deviceName='instr_key_resp_7')
    instr_4_2_sound = sound.Sound(
        'audio/instr_4_2.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_4_2_sound',    name='instr_4_2_sound'
    )
    instr_4_2_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "prac_rest" ---
    # Run 'Begin Experiment' code from code_4
    prac_rest_text = ''
    
    prac_rest_instr = visual.TextStim(win=win, name='prac_rest_instr',
        text='',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_rest_key_resp = keyboard.Keyboard(deviceName='prac_rest_key_resp')
    prac_rest_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='prac_rest_sound',    name='prac_rest_sound'
    )
    prac_rest_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "prac_fixation" ---
    prac_fix = visual.TextStim(win=win, name='prac_fix',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac_exp" ---
    # Run 'Begin Experiment' code from code_2
    count = 0
    tempnum = []
    num = 0
    exp_corr = []
    exp_rt = []
    prac_feedb = ''
    prac_feedbColor = ''
    
    
    prac_stimu = visual.TextStim(win=win, name='prac_stimu',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_blank = visual.TextStim(win=win, name='prac_blank',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    prac_key_resp = keyboard.Keyboard(deviceName='prac_key_resp')
    
    # --- Initialize components for Routine "prac_trial_fb" ---
    prac_feedb_text = visual.TextStim(win=win, name='prac_feedb_text',
        text='',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_feedb_sound = sound.Sound(
        'A', 
        secs=0.65, 
        stereo=True, 
        hamming=True, 
        speaker='prac_feedb_sound',    name='prac_feedb_sound'
    )
    prac_feedb_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "prac_block_fb" ---
    # Run 'Begin Experiment' code from code_5
    block_feedb = ''
    exp_acc = 0
    exp_meanRT = 0
    prac_block_feedb_text = visual.TextStim(win=win, name='prac_block_feedb_text',
        text='',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    prac_block_fb_key_resp = keyboard.Keyboard(deviceName='prac_block_fb_key_resp')
    
    # --- Initialize components for Routine "instr_5" ---
    welpage_exp = visual.TextStim(win=win, name='welpage_exp',
        text='练习结束了。\n\n让我们开始正式实验吧。\n请注意，正式实验中不再提供正误反馈。\n\n记得，在看到数字后立即做出反应，\n试着在保证正确的同时尽可能快地回答。\n\n\n请按空格键继续。',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key_resp_5 = keyboard.Keyboard(deviceName='instr_key_resp_5')
    instr_sound_5 = sound.Sound(
        'audio/instr_5_new.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='instr_sound_5',    name='instr_sound_5'
    )
    instr_sound_5.setVolume(1.0)
    
    # --- Initialize components for Routine "fixation" ---
    fix = visual.TextStim(win=win, name='fix',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "exp" ---
    # Run 'Begin Experiment' code from code
    count = 0
    tempnum = []
    num = 0
    exp_corr = []
    exp_rt = []
    prac_feedb = ''
    prac_feedbColor = ''
    stimu = visual.TextStim(win=win, name='stimu',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    blank = visual.TextStim(win=win, name='blank',
        text=None,
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "block_fb" ---
    # Run 'Begin Experiment' code from code_3
    block_feedb = ''
    blockn = 0
    block_feedb_text = visual.TextStim(win=win, name='block_feedb_text',
        text='',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    block_fb_key_resp = keyboard.Keyboard(deviceName='block_fb_key_resp')
    
    # --- Initialize components for Routine "thanks" ---
    endpage = visual.TextStim(win=win, name='endpage',
        text='实验结束。\n\n感谢您的参与！',
        font='Heiti SC',
        pos=(0, 0), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instr" ---
    # create an object to store info about Routine instr
    instr = data.Routine(
        name='instr',
        components=[welpage, instr_key_resp, instr_sound],
    )
    instr.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp
    instr_key_resp.keys = []
    instr_key_resp.rt = []
    _instr_key_resp_allKeys = []
    instr_sound.setSound('audio/instr.wav', hamming=True)
    instr_sound.setVolume(1.0, log=False)
    instr_sound.seek(0)
    # store start times for instr
    instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr.tStart = globalClock.getTime(format='float')
    instr.status = STARTED
    thisExp.addData('instr.started', instr.tStart)
    instr.maxDuration = None
    # keep track of which components have finished
    instrComponents = instr.components
    for thisComponent in instr.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr" ---
    instr.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage* updates
        
        # if welpage is starting this frame...
        if welpage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage.frameNStart = frameN  # exact frame index
            welpage.tStart = t  # local t and not account for scr refresh
            welpage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage.started')
            # update status
            welpage.status = STARTED
            welpage.setAutoDraw(True)
        
        # if welpage is active this frame...
        if welpage.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp* updates
        waitOnFlip = False
        
        # if instr_key_resp is starting this frame...
        if instr_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp.frameNStart = frameN  # exact frame index
            instr_key_resp.tStart = t  # local t and not account for scr refresh
            instr_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp.started')
            # update status
            instr_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_allKeys.extend(theseKeys)
            if len(_instr_key_resp_allKeys):
                instr_key_resp.keys = _instr_key_resp_allKeys[-1].name  # just the last key pressed
                instr_key_resp.rt = _instr_key_resp_allKeys[-1].rt
                instr_key_resp.duration = _instr_key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_sound* updates
        
        # if instr_sound is starting this frame...
        if instr_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_sound.frameNStart = frameN  # exact frame index
            instr_sound.tStart = t  # local t and not account for scr refresh
            instr_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_sound.started', tThisFlipGlobal)
            # update status
            instr_sound.status = STARTED
            instr_sound.play(when=win)  # sync with win flip
        
        # if instr_sound is stopping this frame...
        if instr_sound.status == STARTED:
            if bool(False) or instr_sound.isFinished:
                # keep track of stop time/frame for later
                instr_sound.tStop = t  # not accounting for scr refresh
                instr_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_sound.stopped')
                # update status
                instr_sound.status = FINISHED
                instr_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr" ---
    for thisComponent in instr.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr
    instr.tStop = globalClock.getTime(format='float')
    instr.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr.stopped', instr.tStop)
    # check responses
    if instr_key_resp.keys in ['', [], None]:  # No response was made
        instr_key_resp.keys = None
    thisExp.addData('instr_key_resp.keys',instr_key_resp.keys)
    if instr_key_resp.keys != None:  # we had a response
        thisExp.addData('instr_key_resp.rt', instr_key_resp.rt)
        thisExp.addData('instr_key_resp.duration', instr_key_resp.duration)
    instr_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_2" ---
    # create an object to store info about Routine instr_2
    instr_2 = data.Routine(
        name='instr_2',
        components=[welpage_2, welpage_2_2, same_e_g, instr_key_resp_2, instr_2_sound],
    )
    instr_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_2
    instr_key_resp_2.keys = []
    instr_key_resp_2.rt = []
    _instr_key_resp_2_allKeys = []
    instr_2_sound.setSound('audio/instr_2.wav', hamming=True)
    instr_2_sound.setVolume(1.0, log=False)
    instr_2_sound.seek(0)
    # store start times for instr_2
    instr_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_2.tStart = globalClock.getTime(format='float')
    instr_2.status = STARTED
    thisExp.addData('instr_2.started', instr_2.tStart)
    instr_2.maxDuration = None
    # keep track of which components have finished
    instr_2Components = instr_2.components
    for thisComponent in instr_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_2" ---
    instr_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_2* updates
        
        # if welpage_2 is starting this frame...
        if welpage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_2.frameNStart = frameN  # exact frame index
            welpage_2.tStart = t  # local t and not account for scr refresh
            welpage_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_2.started')
            # update status
            welpage_2.status = STARTED
            welpage_2.setAutoDraw(True)
        
        # if welpage_2 is active this frame...
        if welpage_2.status == STARTED:
            # update params
            pass
        
        # *welpage_2_2* updates
        
        # if welpage_2_2 is starting this frame...
        if welpage_2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_2_2.frameNStart = frameN  # exact frame index
            welpage_2_2.tStart = t  # local t and not account for scr refresh
            welpage_2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_2_2.started')
            # update status
            welpage_2_2.status = STARTED
            welpage_2_2.setAutoDraw(True)
        
        # if welpage_2_2 is active this frame...
        if welpage_2_2.status == STARTED:
            # update params
            pass
        
        # *same_e_g* updates
        
        # if same_e_g is starting this frame...
        if same_e_g.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            same_e_g.frameNStart = frameN  # exact frame index
            same_e_g.tStart = t  # local t and not account for scr refresh
            same_e_g.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(same_e_g, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'same_e_g.started')
            # update status
            same_e_g.status = STARTED
            same_e_g.setAutoDraw(True)
        
        # if same_e_g is active this frame...
        if same_e_g.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_2* updates
        waitOnFlip = False
        
        # if instr_key_resp_2 is starting this frame...
        if instr_key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_2.frameNStart = frameN  # exact frame index
            instr_key_resp_2.tStart = t  # local t and not account for scr refresh
            instr_key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_2.started')
            # update status
            instr_key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_2.getKeys(keyList=['z'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_2_allKeys.extend(theseKeys)
            if len(_instr_key_resp_2_allKeys):
                instr_key_resp_2.keys = _instr_key_resp_2_allKeys[-1].name  # just the last key pressed
                instr_key_resp_2.rt = _instr_key_resp_2_allKeys[-1].rt
                instr_key_resp_2.duration = _instr_key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_2_sound* updates
        
        # if instr_2_sound is starting this frame...
        if instr_2_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_2_sound.frameNStart = frameN  # exact frame index
            instr_2_sound.tStart = t  # local t and not account for scr refresh
            instr_2_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_2_sound.started', tThisFlipGlobal)
            # update status
            instr_2_sound.status = STARTED
            instr_2_sound.play(when=win)  # sync with win flip
        
        # if instr_2_sound is stopping this frame...
        if instr_2_sound.status == STARTED:
            if bool(False) or instr_2_sound.isFinished:
                # keep track of stop time/frame for later
                instr_2_sound.tStop = t  # not accounting for scr refresh
                instr_2_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_2_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_2_sound.stopped')
                # update status
                instr_2_sound.status = FINISHED
                instr_2_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_2" ---
    for thisComponent in instr_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_2
    instr_2.tStop = globalClock.getTime(format='float')
    instr_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_2.stopped', instr_2.tStop)
    # check responses
    if instr_key_resp_2.keys in ['', [], None]:  # No response was made
        instr_key_resp_2.keys = None
    thisExp.addData('instr_key_resp_2.keys',instr_key_resp_2.keys)
    if instr_key_resp_2.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_2.rt', instr_key_resp_2.rt)
        thisExp.addData('instr_key_resp_2.duration', instr_key_resp_2.duration)
    instr_2_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_3" ---
    # create an object to store info about Routine instr_3
    instr_3 = data.Routine(
        name='instr_3',
        components=[welpage_3, welpage_3_2, different_e_g, instr_key_resp_3, instr_3_sound],
    )
    instr_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_3
    instr_key_resp_3.keys = []
    instr_key_resp_3.rt = []
    _instr_key_resp_3_allKeys = []
    instr_3_sound.setSound('audio/instr_3.wav', hamming=True)
    instr_3_sound.setVolume(1.0, log=False)
    instr_3_sound.seek(0)
    # store start times for instr_3
    instr_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_3.tStart = globalClock.getTime(format='float')
    instr_3.status = STARTED
    thisExp.addData('instr_3.started', instr_3.tStart)
    instr_3.maxDuration = None
    # keep track of which components have finished
    instr_3Components = instr_3.components
    for thisComponent in instr_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_3" ---
    instr_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_3* updates
        
        # if welpage_3 is starting this frame...
        if welpage_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_3.frameNStart = frameN  # exact frame index
            welpage_3.tStart = t  # local t and not account for scr refresh
            welpage_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_3.started')
            # update status
            welpage_3.status = STARTED
            welpage_3.setAutoDraw(True)
        
        # if welpage_3 is active this frame...
        if welpage_3.status == STARTED:
            # update params
            pass
        
        # *welpage_3_2* updates
        
        # if welpage_3_2 is starting this frame...
        if welpage_3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_3_2.frameNStart = frameN  # exact frame index
            welpage_3_2.tStart = t  # local t and not account for scr refresh
            welpage_3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_3_2.started')
            # update status
            welpage_3_2.status = STARTED
            welpage_3_2.setAutoDraw(True)
        
        # if welpage_3_2 is active this frame...
        if welpage_3_2.status == STARTED:
            # update params
            pass
        
        # *different_e_g* updates
        
        # if different_e_g is starting this frame...
        if different_e_g.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            different_e_g.frameNStart = frameN  # exact frame index
            different_e_g.tStart = t  # local t and not account for scr refresh
            different_e_g.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(different_e_g, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'different_e_g.started')
            # update status
            different_e_g.status = STARTED
            different_e_g.setAutoDraw(True)
        
        # if different_e_g is active this frame...
        if different_e_g.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_3* updates
        waitOnFlip = False
        
        # if instr_key_resp_3 is starting this frame...
        if instr_key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_3.frameNStart = frameN  # exact frame index
            instr_key_resp_3.tStart = t  # local t and not account for scr refresh
            instr_key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_3.started')
            # update status
            instr_key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_3.getKeys(keyList=['m'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_3_allKeys.extend(theseKeys)
            if len(_instr_key_resp_3_allKeys):
                instr_key_resp_3.keys = _instr_key_resp_3_allKeys[-1].name  # just the last key pressed
                instr_key_resp_3.rt = _instr_key_resp_3_allKeys[-1].rt
                instr_key_resp_3.duration = _instr_key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_3_sound* updates
        
        # if instr_3_sound is starting this frame...
        if instr_3_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_3_sound.frameNStart = frameN  # exact frame index
            instr_3_sound.tStart = t  # local t and not account for scr refresh
            instr_3_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_3_sound.started', tThisFlipGlobal)
            # update status
            instr_3_sound.status = STARTED
            instr_3_sound.play(when=win)  # sync with win flip
        
        # if instr_3_sound is stopping this frame...
        if instr_3_sound.status == STARTED:
            if bool(False) or instr_3_sound.isFinished:
                # keep track of stop time/frame for later
                instr_3_sound.tStop = t  # not accounting for scr refresh
                instr_3_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_3_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_3_sound.stopped')
                # update status
                instr_3_sound.status = FINISHED
                instr_3_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_3" ---
    for thisComponent in instr_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_3
    instr_3.tStop = globalClock.getTime(format='float')
    instr_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_3.stopped', instr_3.tStop)
    # check responses
    if instr_key_resp_3.keys in ['', [], None]:  # No response was made
        instr_key_resp_3.keys = None
    thisExp.addData('instr_key_resp_3.keys',instr_key_resp_3.keys)
    if instr_key_resp_3.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_3.rt', instr_key_resp_3.rt)
        thisExp.addData('instr_key_resp_3.duration', instr_key_resp_3.duration)
    instr_3_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_3_1" ---
    # create an object to store info about Routine instr_3_1
    instr_3_1 = data.Routine(
        name='instr_3_1',
        components=[welpage_3_1, instr_key_resp_3_1, instr_3_1_sound],
    )
    instr_3_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_3_1
    instr_key_resp_3_1.keys = []
    instr_key_resp_3_1.rt = []
    _instr_key_resp_3_1_allKeys = []
    instr_3_1_sound.setSound('audio/instr_3_1.wav', hamming=True)
    instr_3_1_sound.setVolume(1.0, log=False)
    instr_3_1_sound.seek(0)
    # store start times for instr_3_1
    instr_3_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_3_1.tStart = globalClock.getTime(format='float')
    instr_3_1.status = STARTED
    thisExp.addData('instr_3_1.started', instr_3_1.tStart)
    instr_3_1.maxDuration = None
    # keep track of which components have finished
    instr_3_1Components = instr_3_1.components
    for thisComponent in instr_3_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_3_1" ---
    instr_3_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_3_1* updates
        
        # if welpage_3_1 is starting this frame...
        if welpage_3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_3_1.frameNStart = frameN  # exact frame index
            welpage_3_1.tStart = t  # local t and not account for scr refresh
            welpage_3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_3_1.started')
            # update status
            welpage_3_1.status = STARTED
            welpage_3_1.setAutoDraw(True)
        
        # if welpage_3_1 is active this frame...
        if welpage_3_1.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_3_1* updates
        waitOnFlip = False
        
        # if instr_key_resp_3_1 is starting this frame...
        if instr_key_resp_3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_3_1.frameNStart = frameN  # exact frame index
            instr_key_resp_3_1.tStart = t  # local t and not account for scr refresh
            instr_key_resp_3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_3_1.started')
            # update status
            instr_key_resp_3_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_3_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_3_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_3_1.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_3_1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_3_1_allKeys.extend(theseKeys)
            if len(_instr_key_resp_3_1_allKeys):
                instr_key_resp_3_1.keys = _instr_key_resp_3_1_allKeys[-1].name  # just the last key pressed
                instr_key_resp_3_1.rt = _instr_key_resp_3_1_allKeys[-1].rt
                instr_key_resp_3_1.duration = _instr_key_resp_3_1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_3_1_sound* updates
        
        # if instr_3_1_sound is starting this frame...
        if instr_3_1_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_3_1_sound.frameNStart = frameN  # exact frame index
            instr_3_1_sound.tStart = t  # local t and not account for scr refresh
            instr_3_1_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_3_1_sound.started', tThisFlipGlobal)
            # update status
            instr_3_1_sound.status = STARTED
            instr_3_1_sound.play(when=win)  # sync with win flip
        
        # if instr_3_1_sound is stopping this frame...
        if instr_3_1_sound.status == STARTED:
            if bool(False) or instr_3_1_sound.isFinished:
                # keep track of stop time/frame for later
                instr_3_1_sound.tStop = t  # not accounting for scr refresh
                instr_3_1_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_3_1_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_3_1_sound.stopped')
                # update status
                instr_3_1_sound.status = FINISHED
                instr_3_1_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_3_1,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_3_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_3_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_3_1" ---
    for thisComponent in instr_3_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_3_1
    instr_3_1.tStop = globalClock.getTime(format='float')
    instr_3_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_3_1.stopped', instr_3_1.tStop)
    # check responses
    if instr_key_resp_3_1.keys in ['', [], None]:  # No response was made
        instr_key_resp_3_1.keys = None
    thisExp.addData('instr_key_resp_3_1.keys',instr_key_resp_3_1.keys)
    if instr_key_resp_3_1.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_3_1.rt', instr_key_resp_3_1.rt)
        thisExp.addData('instr_key_resp_3_1.duration', instr_key_resp_3_1.duration)
    instr_3_1_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_3_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_judgement = data.TrialHandler2(
        name='practice_judgement',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_judgement)  # add the loop to the experiment
    thisPractice_judgement = practice_judgement.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_judgement.rgb)
    if thisPractice_judgement != None:
        for paramName in thisPractice_judgement:
            globals()[paramName] = thisPractice_judgement[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_judgement in practice_judgement:
        practice_judgement.status = STARTED
        if hasattr(thisPractice_judgement, 'status'):
            thisPractice_judgement.status = STARTED
        currentLoop = practice_judgement
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_judgement.rgb)
        if thisPractice_judgement != None:
            for paramName in thisPractice_judgement:
                globals()[paramName] = thisPractice_judgement[paramName]
        
        # set up handler to look after randomisation of conditions etc
        instr_loop = data.TrialHandler2(
            name='instr_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(instr_loop)  # add the loop to the experiment
        thisInstr_loop = instr_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInstr_loop.rgb)
        if thisInstr_loop != None:
            for paramName in thisInstr_loop:
                globals()[paramName] = thisInstr_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisInstr_loop in instr_loop:
            instr_loop.status = STARTED
            if hasattr(thisInstr_loop, 'status'):
                thisInstr_loop.status = STARTED
            currentLoop = instr_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisInstr_loop.rgb)
            if thisInstr_loop != None:
                for paramName in thisInstr_loop:
                    globals()[paramName] = thisInstr_loop[paramName]
            
            # --- Prepare to start Routine "instr_3_2" ---
            # create an object to store info about Routine instr_3_2
            instr_3_2 = data.Routine(
                name='instr_3_2',
                components=[text_45, text_9, text, instr_3_2_sound],
            )
            instr_3_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            instr_3_2_sound.setSound('audio/instr_3_2.wav', secs=6, hamming=True)
            instr_3_2_sound.setVolume(1.0, log=False)
            instr_3_2_sound.seek(0)
            # store start times for instr_3_2
            instr_3_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_2.tStart = globalClock.getTime(format='float')
            instr_3_2.status = STARTED
            thisExp.addData('instr_3_2.started', instr_3_2.tStart)
            instr_3_2.maxDuration = None
            # keep track of which components have finished
            instr_3_2Components = instr_3_2.components
            for thisComponent in instr_3_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_2" ---
            instr_3_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 6.0:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_45* updates
                
                # if text_45 is starting this frame...
                if text_45.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_45.frameNStart = frameN  # exact frame index
                    text_45.tStart = t  # local t and not account for scr refresh
                    text_45.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_45, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_45.started')
                    # update status
                    text_45.status = STARTED
                    text_45.setAutoDraw(True)
                
                # if text_45 is active this frame...
                if text_45.status == STARTED:
                    # update params
                    pass
                
                # if text_45 is stopping this frame...
                if text_45.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_45.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        text_45.tStop = t  # not accounting for scr refresh
                        text_45.tStopRefresh = tThisFlipGlobal  # on global time
                        text_45.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_45.stopped')
                        # update status
                        text_45.status = FINISHED
                        text_45.setAutoDraw(False)
                
                # *text_9* updates
                
                # if text_9 is starting this frame...
                if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_9.frameNStart = frameN  # exact frame index
                    text_9.tStart = t  # local t and not account for scr refresh
                    text_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_9.started')
                    # update status
                    text_9.status = STARTED
                    text_9.setAutoDraw(True)
                
                # if text_9 is active this frame...
                if text_9.status == STARTED:
                    # update params
                    pass
                
                # if text_9 is stopping this frame...
                if text_9.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_9.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        text_9.tStop = t  # not accounting for scr refresh
                        text_9.tStopRefresh = tThisFlipGlobal  # on global time
                        text_9.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_9.stopped')
                        # update status
                        text_9.status = FINISHED
                        text_9.setAutoDraw(False)
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.tStopRefresh = tThisFlipGlobal  # on global time
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # *instr_3_2_sound* updates
                
                # if instr_3_2_sound is starting this frame...
                if instr_3_2_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_2_sound.frameNStart = frameN  # exact frame index
                    instr_3_2_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_2_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_2_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_2_sound.status = STARTED
                    instr_3_2_sound.play(when=win)  # sync with win flip
                
                # if instr_3_2_sound is stopping this frame...
                if instr_3_2_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > instr_3_2_sound.tStartRefresh + 6-frameTolerance or instr_3_2_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_2_sound.tStop = t  # not accounting for scr refresh
                        instr_3_2_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_2_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_2_sound.stopped')
                        # update status
                        instr_3_2_sound.status = FINISHED
                        instr_3_2_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_2,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_2" ---
            for thisComponent in instr_3_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_2
            instr_3_2.tStop = globalClock.getTime(format='float')
            instr_3_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_2.stopped', instr_3_2.tStop)
            instr_3_2_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if instr_3_2.maxDurationReached:
                routineTimer.addTime(-instr_3_2.maxDuration)
            elif instr_3_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-6.000000)
            
            # --- Prepare to start Routine "instr_3_3" ---
            # create an object to store info about Routine instr_3_3
            instr_3_3 = data.Routine(
                name='instr_3_3',
                components=[text_2, key_resp_2, instr_3_3_sound],
            )
            instr_3_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_2
            key_resp_2.keys = []
            key_resp_2.rt = []
            _key_resp_2_allKeys = []
            instr_3_3_sound.setSound('audio/instr_3_3.wav', hamming=True)
            instr_3_3_sound.setVolume(1.0, log=False)
            instr_3_3_sound.seek(0)
            # store start times for instr_3_3
            instr_3_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_3.tStart = globalClock.getTime(format='float')
            instr_3_3.status = STARTED
            thisExp.addData('instr_3_3.started', instr_3_3.tStart)
            instr_3_3.maxDuration = None
            # keep track of which components have finished
            instr_3_3Components = instr_3_3.components
            for thisComponent in instr_3_3.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_3" ---
            instr_3_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_2* updates
                
                # if text_2 is starting this frame...
                if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_2.frameNStart = frameN  # exact frame index
                    text_2.tStart = t  # local t and not account for scr refresh
                    text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.started')
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_2* updates
                waitOnFlip = False
                
                # if key_resp_2 is starting this frame...
                if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_2.frameNStart = frameN  # exact frame index
                    key_resp_2.tStart = t  # local t and not account for scr refresh
                    key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_2.started')
                    # update status
                    key_resp_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_2_allKeys.extend(theseKeys)
                    if len(_key_resp_2_allKeys):
                        key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                        key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                        key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_3_sound* updates
                
                # if instr_3_3_sound is starting this frame...
                if instr_3_3_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_3_sound.frameNStart = frameN  # exact frame index
                    instr_3_3_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_3_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_3_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_3_sound.status = STARTED
                    instr_3_3_sound.play(when=win)  # sync with win flip
                
                # if instr_3_3_sound is stopping this frame...
                if instr_3_3_sound.status == STARTED:
                    if bool(False) or instr_3_3_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_3_sound.tStop = t  # not accounting for scr refresh
                        instr_3_3_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_3_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_3_sound.stopped')
                        # update status
                        instr_3_3_sound.status = FINISHED
                        instr_3_3_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_3,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_3" ---
            for thisComponent in instr_3_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_3
            instr_3_3.tStop = globalClock.getTime(format='float')
            instr_3_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_3.stopped', instr_3_3.tStop)
            # check responses
            if key_resp_2.keys in ['', [], None]:  # No response was made
                key_resp_2.keys = None
            instr_loop.addData('key_resp_2.keys',key_resp_2.keys)
            if key_resp_2.keys != None:  # we had a response
                instr_loop.addData('key_resp_2.rt', key_resp_2.rt)
                instr_loop.addData('key_resp_2.duration', key_resp_2.duration)
            instr_3_3_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_3" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_4" ---
            # create an object to store info about Routine instr_3_4
            instr_3_4 = data.Routine(
                name='instr_3_4',
                components=[text_10, text_3, instr_3_4_sound],
            )
            instr_3_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            instr_3_4_sound.setSound('audio/instr_3_2.wav', secs=6, hamming=True)
            instr_3_4_sound.setVolume(1.0, log=False)
            instr_3_4_sound.seek(0)
            # store start times for instr_3_4
            instr_3_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_4.tStart = globalClock.getTime(format='float')
            instr_3_4.status = STARTED
            thisExp.addData('instr_3_4.started', instr_3_4.tStart)
            instr_3_4.maxDuration = None
            # keep track of which components have finished
            instr_3_4Components = instr_3_4.components
            for thisComponent in instr_3_4.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_4" ---
            instr_3_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 6.0:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_10* updates
                
                # if text_10 is starting this frame...
                if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_10.frameNStart = frameN  # exact frame index
                    text_10.tStart = t  # local t and not account for scr refresh
                    text_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_10.started')
                    # update status
                    text_10.status = STARTED
                    text_10.setAutoDraw(True)
                
                # if text_10 is active this frame...
                if text_10.status == STARTED:
                    # update params
                    pass
                
                # if text_10 is stopping this frame...
                if text_10.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_10.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        text_10.tStop = t  # not accounting for scr refresh
                        text_10.tStopRefresh = tThisFlipGlobal  # on global time
                        text_10.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_10.stopped')
                        # update status
                        text_10.status = FINISHED
                        text_10.setAutoDraw(False)
                
                # *text_3* updates
                
                # if text_3 is starting this frame...
                if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_3.frameNStart = frameN  # exact frame index
                    text_3.tStart = t  # local t and not account for scr refresh
                    text_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.started')
                    # update status
                    text_3.status = STARTED
                    text_3.setAutoDraw(True)
                
                # if text_3 is active this frame...
                if text_3.status == STARTED:
                    # update params
                    pass
                
                # if text_3 is stopping this frame...
                if text_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_3.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        text_3.tStop = t  # not accounting for scr refresh
                        text_3.tStopRefresh = tThisFlipGlobal  # on global time
                        text_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_3.stopped')
                        # update status
                        text_3.status = FINISHED
                        text_3.setAutoDraw(False)
                
                # *instr_3_4_sound* updates
                
                # if instr_3_4_sound is starting this frame...
                if instr_3_4_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_4_sound.frameNStart = frameN  # exact frame index
                    instr_3_4_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_4_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_4_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_4_sound.status = STARTED
                    instr_3_4_sound.play(when=win)  # sync with win flip
                
                # if instr_3_4_sound is stopping this frame...
                if instr_3_4_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > instr_3_4_sound.tStartRefresh + 6-frameTolerance or instr_3_4_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_4_sound.tStop = t  # not accounting for scr refresh
                        instr_3_4_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_4_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_4_sound.stopped')
                        # update status
                        instr_3_4_sound.status = FINISHED
                        instr_3_4_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_4,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_4" ---
            for thisComponent in instr_3_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_4
            instr_3_4.tStop = globalClock.getTime(format='float')
            instr_3_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_4.stopped', instr_3_4.tStop)
            instr_3_4_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if instr_3_4.maxDurationReached:
                routineTimer.addTime(-instr_3_4.maxDuration)
            elif instr_3_4.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-6.000000)
            
            # --- Prepare to start Routine "instr_3_5" ---
            # create an object to store info about Routine instr_3_5
            instr_3_5 = data.Routine(
                name='instr_3_5',
                components=[text_4, key_resp_3, instr_3_5_sound],
            )
            instr_3_5.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_3
            key_resp_3.keys = []
            key_resp_3.rt = []
            _key_resp_3_allKeys = []
            instr_3_5_sound.setSound('audio/instr_3_5.wav', hamming=True)
            instr_3_5_sound.setVolume(1.0, log=False)
            instr_3_5_sound.seek(0)
            # store start times for instr_3_5
            instr_3_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_5.tStart = globalClock.getTime(format='float')
            instr_3_5.status = STARTED
            thisExp.addData('instr_3_5.started', instr_3_5.tStart)
            instr_3_5.maxDuration = None
            # keep track of which components have finished
            instr_3_5Components = instr_3_5.components
            for thisComponent in instr_3_5.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_5" ---
            instr_3_5.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_4* updates
                
                # if text_4 is starting this frame...
                if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_4.frameNStart = frameN  # exact frame index
                    text_4.tStart = t  # local t and not account for scr refresh
                    text_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.started')
                    # update status
                    text_4.status = STARTED
                    text_4.setAutoDraw(True)
                
                # if text_4 is active this frame...
                if text_4.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_3* updates
                waitOnFlip = False
                
                # if key_resp_3 is starting this frame...
                if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_3.frameNStart = frameN  # exact frame index
                    key_resp_3.tStart = t  # local t and not account for scr refresh
                    key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_3.started')
                    # update status
                    key_resp_3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_3.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_3_allKeys.extend(theseKeys)
                    if len(_key_resp_3_allKeys):
                        key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                        key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                        key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_5_sound* updates
                
                # if instr_3_5_sound is starting this frame...
                if instr_3_5_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_5_sound.frameNStart = frameN  # exact frame index
                    instr_3_5_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_5_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_5_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_5_sound.status = STARTED
                    instr_3_5_sound.play(when=win)  # sync with win flip
                
                # if instr_3_5_sound is stopping this frame...
                if instr_3_5_sound.status == STARTED:
                    if bool(False) or instr_3_5_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_5_sound.tStop = t  # not accounting for scr refresh
                        instr_3_5_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_5_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_5_sound.stopped')
                        # update status
                        instr_3_5_sound.status = FINISHED
                        instr_3_5_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_5,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_5.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_5.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_5" ---
            for thisComponent in instr_3_5.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_5
            instr_3_5.tStop = globalClock.getTime(format='float')
            instr_3_5.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_5.stopped', instr_3_5.tStop)
            # check responses
            if key_resp_3.keys in ['', [], None]:  # No response was made
                key_resp_3.keys = None
            instr_loop.addData('key_resp_3.keys',key_resp_3.keys)
            if key_resp_3.keys != None:  # we had a response
                instr_loop.addData('key_resp_3.rt', key_resp_3.rt)
                instr_loop.addData('key_resp_3.duration', key_resp_3.duration)
            instr_3_5_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_5" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_6" ---
            # create an object to store info about Routine instr_3_6
            instr_3_6 = data.Routine(
                name='instr_3_6',
                components=[text_12, text_5, text_11, key_resp_4, instr_3_6_sound],
            )
            instr_3_6.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_4
            key_resp_4.keys = []
            key_resp_4.rt = []
            _key_resp_4_allKeys = []
            instr_3_6_sound.setSound('audio/instr_3_2.wav', hamming=True)
            instr_3_6_sound.setVolume(1.0, log=False)
            instr_3_6_sound.seek(0)
            # store start times for instr_3_6
            instr_3_6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_6.tStart = globalClock.getTime(format='float')
            instr_3_6.status = STARTED
            thisExp.addData('instr_3_6.started', instr_3_6.tStart)
            instr_3_6.maxDuration = None
            # keep track of which components have finished
            instr_3_6Components = instr_3_6.components
            for thisComponent in instr_3_6.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_6" ---
            instr_3_6.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_12* updates
                
                # if text_12 is starting this frame...
                if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_12.frameNStart = frameN  # exact frame index
                    text_12.tStart = t  # local t and not account for scr refresh
                    text_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_12.started')
                    # update status
                    text_12.status = STARTED
                    text_12.setAutoDraw(True)
                
                # if text_12 is active this frame...
                if text_12.status == STARTED:
                    # update params
                    pass
                
                # *text_5* updates
                
                # if text_5 is starting this frame...
                if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_5.frameNStart = frameN  # exact frame index
                    text_5.tStart = t  # local t and not account for scr refresh
                    text_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_5.started')
                    # update status
                    text_5.status = STARTED
                    text_5.setAutoDraw(True)
                
                # if text_5 is active this frame...
                if text_5.status == STARTED:
                    # update params
                    pass
                
                # *text_11* updates
                
                # if text_11 is starting this frame...
                if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_11.frameNStart = frameN  # exact frame index
                    text_11.tStart = t  # local t and not account for scr refresh
                    text_11.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_11.started')
                    # update status
                    text_11.status = STARTED
                    text_11.setAutoDraw(True)
                
                # if text_11 is active this frame...
                if text_11.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_4* updates
                waitOnFlip = False
                
                # if key_resp_4 is starting this frame...
                if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_4.frameNStart = frameN  # exact frame index
                    key_resp_4.tStart = t  # local t and not account for scr refresh
                    key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_4.started')
                    # update status
                    key_resp_4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_4.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_4.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_4_allKeys.extend(theseKeys)
                    if len(_key_resp_4_allKeys):
                        key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                        key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                        key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_6_sound* updates
                
                # if instr_3_6_sound is starting this frame...
                if instr_3_6_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_6_sound.frameNStart = frameN  # exact frame index
                    instr_3_6_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_6_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_6_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_6_sound.status = STARTED
                    instr_3_6_sound.play(when=win)  # sync with win flip
                
                # if instr_3_6_sound is stopping this frame...
                if instr_3_6_sound.status == STARTED:
                    if bool(False) or instr_3_6_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_6_sound.tStop = t  # not accounting for scr refresh
                        instr_3_6_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_6_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_6_sound.stopped')
                        # update status
                        instr_3_6_sound.status = FINISHED
                        instr_3_6_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_6,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_6.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_6.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_6" ---
            for thisComponent in instr_3_6.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_6
            instr_3_6.tStop = globalClock.getTime(format='float')
            instr_3_6.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_6.stopped', instr_3_6.tStop)
            # check responses
            if key_resp_4.keys in ['', [], None]:  # No response was made
                key_resp_4.keys = None
            instr_loop.addData('key_resp_4.keys',key_resp_4.keys)
            if key_resp_4.keys != None:  # we had a response
                instr_loop.addData('key_resp_4.rt', key_resp_4.rt)
                instr_loop.addData('key_resp_4.duration', key_resp_4.duration)
            instr_3_6_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_6" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_7" ---
            # create an object to store info about Routine instr_3_7
            instr_3_7 = data.Routine(
                name='instr_3_7',
                components=[text_6, text_7, text_13, text_8, text_14, key_resp_5, instr_3_7_sound],
            )
            instr_3_7.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_5
            key_resp_5.keys = []
            key_resp_5.rt = []
            _key_resp_5_allKeys = []
            instr_3_7_sound.setSound('audio/instr_3_7.wav', hamming=True)
            instr_3_7_sound.setVolume(1.0, log=False)
            instr_3_7_sound.seek(0)
            # store start times for instr_3_7
            instr_3_7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_7.tStart = globalClock.getTime(format='float')
            instr_3_7.status = STARTED
            thisExp.addData('instr_3_7.started', instr_3_7.tStart)
            instr_3_7.maxDuration = None
            # keep track of which components have finished
            instr_3_7Components = instr_3_7.components
            for thisComponent in instr_3_7.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_7" ---
            instr_3_7.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_6* updates
                
                # if text_6 is starting this frame...
                if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_6.frameNStart = frameN  # exact frame index
                    text_6.tStart = t  # local t and not account for scr refresh
                    text_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_6.started')
                    # update status
                    text_6.status = STARTED
                    text_6.setAutoDraw(True)
                
                # if text_6 is active this frame...
                if text_6.status == STARTED:
                    # update params
                    pass
                
                # *text_7* updates
                
                # if text_7 is starting this frame...
                if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_7.frameNStart = frameN  # exact frame index
                    text_7.tStart = t  # local t and not account for scr refresh
                    text_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_7.started')
                    # update status
                    text_7.status = STARTED
                    text_7.setAutoDraw(True)
                
                # if text_7 is active this frame...
                if text_7.status == STARTED:
                    # update params
                    pass
                
                # *text_13* updates
                
                # if text_13 is starting this frame...
                if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_13.frameNStart = frameN  # exact frame index
                    text_13.tStart = t  # local t and not account for scr refresh
                    text_13.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_13.started')
                    # update status
                    text_13.status = STARTED
                    text_13.setAutoDraw(True)
                
                # if text_13 is active this frame...
                if text_13.status == STARTED:
                    # update params
                    pass
                
                # *text_8* updates
                
                # if text_8 is starting this frame...
                if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_8.frameNStart = frameN  # exact frame index
                    text_8.tStart = t  # local t and not account for scr refresh
                    text_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.started')
                    # update status
                    text_8.status = STARTED
                    text_8.setAutoDraw(True)
                
                # if text_8 is active this frame...
                if text_8.status == STARTED:
                    # update params
                    pass
                
                # *text_14* updates
                
                # if text_14 is starting this frame...
                if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_14.frameNStart = frameN  # exact frame index
                    text_14.tStart = t  # local t and not account for scr refresh
                    text_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_14.started')
                    # update status
                    text_14.status = STARTED
                    text_14.setAutoDraw(True)
                
                # if text_14 is active this frame...
                if text_14.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_5* updates
                waitOnFlip = False
                
                # if key_resp_5 is starting this frame...
                if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_5.frameNStart = frameN  # exact frame index
                    key_resp_5.tStart = t  # local t and not account for scr refresh
                    key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_5.started')
                    # update status
                    key_resp_5.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_5.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_5_allKeys.extend(theseKeys)
                    if len(_key_resp_5_allKeys):
                        key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                        key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                        key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_7_sound* updates
                
                # if instr_3_7_sound is starting this frame...
                if instr_3_7_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_7_sound.frameNStart = frameN  # exact frame index
                    instr_3_7_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_7_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_7_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_7_sound.status = STARTED
                    instr_3_7_sound.play(when=win)  # sync with win flip
                
                # if instr_3_7_sound is stopping this frame...
                if instr_3_7_sound.status == STARTED:
                    if bool(False) or instr_3_7_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_7_sound.tStop = t  # not accounting for scr refresh
                        instr_3_7_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_7_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_7_sound.stopped')
                        # update status
                        instr_3_7_sound.status = FINISHED
                        instr_3_7_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_7,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_7.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_7.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_7" ---
            for thisComponent in instr_3_7.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_7
            instr_3_7.tStop = globalClock.getTime(format='float')
            instr_3_7.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_7.stopped', instr_3_7.tStop)
            # check responses
            if key_resp_5.keys in ['', [], None]:  # No response was made
                key_resp_5.keys = None
            instr_loop.addData('key_resp_5.keys',key_resp_5.keys)
            if key_resp_5.keys != None:  # we had a response
                instr_loop.addData('key_resp_5.rt', key_resp_5.rt)
                instr_loop.addData('key_resp_5.duration', key_resp_5.duration)
            instr_3_7_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_7" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_8" ---
            # create an object to store info about Routine instr_3_8
            instr_3_8 = data.Routine(
                name='instr_3_8',
                components=[text_15, text_16, text_17, key_resp_6, instr_3_8_sound],
            )
            instr_3_8.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_6
            key_resp_6.keys = []
            key_resp_6.rt = []
            _key_resp_6_allKeys = []
            instr_3_8_sound.setSound('audio/instr_3_2.wav', hamming=True)
            instr_3_8_sound.setVolume(1.0, log=False)
            instr_3_8_sound.seek(0)
            # store start times for instr_3_8
            instr_3_8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_8.tStart = globalClock.getTime(format='float')
            instr_3_8.status = STARTED
            thisExp.addData('instr_3_8.started', instr_3_8.tStart)
            instr_3_8.maxDuration = None
            # keep track of which components have finished
            instr_3_8Components = instr_3_8.components
            for thisComponent in instr_3_8.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_8" ---
            instr_3_8.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_15* updates
                
                # if text_15 is starting this frame...
                if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_15.frameNStart = frameN  # exact frame index
                    text_15.tStart = t  # local t and not account for scr refresh
                    text_15.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_15.started')
                    # update status
                    text_15.status = STARTED
                    text_15.setAutoDraw(True)
                
                # if text_15 is active this frame...
                if text_15.status == STARTED:
                    # update params
                    pass
                
                # *text_16* updates
                
                # if text_16 is starting this frame...
                if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_16.frameNStart = frameN  # exact frame index
                    text_16.tStart = t  # local t and not account for scr refresh
                    text_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_16.started')
                    # update status
                    text_16.status = STARTED
                    text_16.setAutoDraw(True)
                
                # if text_16 is active this frame...
                if text_16.status == STARTED:
                    # update params
                    pass
                
                # *text_17* updates
                
                # if text_17 is starting this frame...
                if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_17.frameNStart = frameN  # exact frame index
                    text_17.tStart = t  # local t and not account for scr refresh
                    text_17.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_17.started')
                    # update status
                    text_17.status = STARTED
                    text_17.setAutoDraw(True)
                
                # if text_17 is active this frame...
                if text_17.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_6* updates
                waitOnFlip = False
                
                # if key_resp_6 is starting this frame...
                if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_6.frameNStart = frameN  # exact frame index
                    key_resp_6.tStart = t  # local t and not account for scr refresh
                    key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_6.started')
                    # update status
                    key_resp_6.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_6.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_6.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_6_allKeys.extend(theseKeys)
                    if len(_key_resp_6_allKeys):
                        key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                        key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                        key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_8_sound* updates
                
                # if instr_3_8_sound is starting this frame...
                if instr_3_8_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_8_sound.frameNStart = frameN  # exact frame index
                    instr_3_8_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_8_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_8_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_8_sound.status = STARTED
                    instr_3_8_sound.play(when=win)  # sync with win flip
                
                # if instr_3_8_sound is stopping this frame...
                if instr_3_8_sound.status == STARTED:
                    if bool(False) or instr_3_8_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_8_sound.tStop = t  # not accounting for scr refresh
                        instr_3_8_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_8_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_8_sound.stopped')
                        # update status
                        instr_3_8_sound.status = FINISHED
                        instr_3_8_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_8,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_8.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_8.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_8" ---
            for thisComponent in instr_3_8.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_8
            instr_3_8.tStop = globalClock.getTime(format='float')
            instr_3_8.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_8.stopped', instr_3_8.tStop)
            # check responses
            if key_resp_6.keys in ['', [], None]:  # No response was made
                key_resp_6.keys = None
            instr_loop.addData('key_resp_6.keys',key_resp_6.keys)
            if key_resp_6.keys != None:  # we had a response
                instr_loop.addData('key_resp_6.rt', key_resp_6.rt)
                instr_loop.addData('key_resp_6.duration', key_resp_6.duration)
            instr_3_8_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_8" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_9" ---
            # create an object to store info about Routine instr_3_9
            instr_3_9 = data.Routine(
                name='instr_3_9',
                components=[text_18, text_19, text_23, text_20, text_21, text_22, key_resp_7, instr_3_9_sound],
            )
            instr_3_9.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_7
            key_resp_7.keys = []
            key_resp_7.rt = []
            _key_resp_7_allKeys = []
            instr_3_9_sound.setSound('audio/instr_3_9.wav', hamming=True)
            instr_3_9_sound.setVolume(1.0, log=False)
            instr_3_9_sound.seek(0)
            # store start times for instr_3_9
            instr_3_9.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_9.tStart = globalClock.getTime(format='float')
            instr_3_9.status = STARTED
            thisExp.addData('instr_3_9.started', instr_3_9.tStart)
            instr_3_9.maxDuration = None
            # keep track of which components have finished
            instr_3_9Components = instr_3_9.components
            for thisComponent in instr_3_9.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_9" ---
            instr_3_9.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_18* updates
                
                # if text_18 is starting this frame...
                if text_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_18.frameNStart = frameN  # exact frame index
                    text_18.tStart = t  # local t and not account for scr refresh
                    text_18.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_18, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_18.started')
                    # update status
                    text_18.status = STARTED
                    text_18.setAutoDraw(True)
                
                # if text_18 is active this frame...
                if text_18.status == STARTED:
                    # update params
                    pass
                
                # *text_19* updates
                
                # if text_19 is starting this frame...
                if text_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_19.frameNStart = frameN  # exact frame index
                    text_19.tStart = t  # local t and not account for scr refresh
                    text_19.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_19, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_19.started')
                    # update status
                    text_19.status = STARTED
                    text_19.setAutoDraw(True)
                
                # if text_19 is active this frame...
                if text_19.status == STARTED:
                    # update params
                    pass
                
                # *text_23* updates
                
                # if text_23 is starting this frame...
                if text_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_23.frameNStart = frameN  # exact frame index
                    text_23.tStart = t  # local t and not account for scr refresh
                    text_23.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_23, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_23.started')
                    # update status
                    text_23.status = STARTED
                    text_23.setAutoDraw(True)
                
                # if text_23 is active this frame...
                if text_23.status == STARTED:
                    # update params
                    pass
                
                # *text_20* updates
                
                # if text_20 is starting this frame...
                if text_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_20.frameNStart = frameN  # exact frame index
                    text_20.tStart = t  # local t and not account for scr refresh
                    text_20.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_20, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_20.started')
                    # update status
                    text_20.status = STARTED
                    text_20.setAutoDraw(True)
                
                # if text_20 is active this frame...
                if text_20.status == STARTED:
                    # update params
                    pass
                
                # *text_21* updates
                
                # if text_21 is starting this frame...
                if text_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_21.frameNStart = frameN  # exact frame index
                    text_21.tStart = t  # local t and not account for scr refresh
                    text_21.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_21, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_21.started')
                    # update status
                    text_21.status = STARTED
                    text_21.setAutoDraw(True)
                
                # if text_21 is active this frame...
                if text_21.status == STARTED:
                    # update params
                    pass
                
                # *text_22* updates
                
                # if text_22 is starting this frame...
                if text_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_22.frameNStart = frameN  # exact frame index
                    text_22.tStart = t  # local t and not account for scr refresh
                    text_22.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_22, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_22.started')
                    # update status
                    text_22.status = STARTED
                    text_22.setAutoDraw(True)
                
                # if text_22 is active this frame...
                if text_22.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_7* updates
                waitOnFlip = False
                
                # if key_resp_7 is starting this frame...
                if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_7.frameNStart = frameN  # exact frame index
                    key_resp_7.tStart = t  # local t and not account for scr refresh
                    key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_7.started')
                    # update status
                    key_resp_7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_7.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_7_allKeys.extend(theseKeys)
                    if len(_key_resp_7_allKeys):
                        key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                        key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                        key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_9_sound* updates
                
                # if instr_3_9_sound is starting this frame...
                if instr_3_9_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_9_sound.frameNStart = frameN  # exact frame index
                    instr_3_9_sound.tStart = t  # local t and not account for scr refresh
                    instr_3_9_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_9_sound.started', tThisFlipGlobal)
                    # update status
                    instr_3_9_sound.status = STARTED
                    instr_3_9_sound.play(when=win)  # sync with win flip
                
                # if instr_3_9_sound is stopping this frame...
                if instr_3_9_sound.status == STARTED:
                    if bool(False) or instr_3_9_sound.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_9_sound.tStop = t  # not accounting for scr refresh
                        instr_3_9_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_9_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_9_sound.stopped')
                        # update status
                        instr_3_9_sound.status = FINISHED
                        instr_3_9_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_9,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_9.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_9.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_9" ---
            for thisComponent in instr_3_9.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_9
            instr_3_9.tStop = globalClock.getTime(format='float')
            instr_3_9.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_9.stopped', instr_3_9.tStop)
            # check responses
            if key_resp_7.keys in ['', [], None]:  # No response was made
                key_resp_7.keys = None
            instr_loop.addData('key_resp_7.keys',key_resp_7.keys)
            if key_resp_7.keys != None:  # we had a response
                instr_loop.addData('key_resp_7.rt', key_resp_7.rt)
                instr_loop.addData('key_resp_7.duration', key_resp_7.duration)
            instr_3_9_sound.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_9" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_10" ---
            # create an object to store info about Routine instr_3_10
            instr_3_10 = data.Routine(
                name='instr_3_10',
                components=[text_24, text_25, text_26, key_resp_8, instr_3_8_sound_2],
            )
            instr_3_10.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_8
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            instr_3_8_sound_2.setSound('audio/instr_3_2.wav', hamming=True)
            instr_3_8_sound_2.setVolume(1.0, log=False)
            instr_3_8_sound_2.seek(0)
            # store start times for instr_3_10
            instr_3_10.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_10.tStart = globalClock.getTime(format='float')
            instr_3_10.status = STARTED
            thisExp.addData('instr_3_10.started', instr_3_10.tStart)
            instr_3_10.maxDuration = None
            # keep track of which components have finished
            instr_3_10Components = instr_3_10.components
            for thisComponent in instr_3_10.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_10" ---
            instr_3_10.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_24* updates
                
                # if text_24 is starting this frame...
                if text_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_24.frameNStart = frameN  # exact frame index
                    text_24.tStart = t  # local t and not account for scr refresh
                    text_24.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_24, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_24.started')
                    # update status
                    text_24.status = STARTED
                    text_24.setAutoDraw(True)
                
                # if text_24 is active this frame...
                if text_24.status == STARTED:
                    # update params
                    pass
                
                # *text_25* updates
                
                # if text_25 is starting this frame...
                if text_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_25.frameNStart = frameN  # exact frame index
                    text_25.tStart = t  # local t and not account for scr refresh
                    text_25.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_25, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_25.started')
                    # update status
                    text_25.status = STARTED
                    text_25.setAutoDraw(True)
                
                # if text_25 is active this frame...
                if text_25.status == STARTED:
                    # update params
                    pass
                
                # *text_26* updates
                
                # if text_26 is starting this frame...
                if text_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_26.frameNStart = frameN  # exact frame index
                    text_26.tStart = t  # local t and not account for scr refresh
                    text_26.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_26, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_26.started')
                    # update status
                    text_26.status = STARTED
                    text_26.setAutoDraw(True)
                
                # if text_26 is active this frame...
                if text_26.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_8_sound_2* updates
                
                # if instr_3_8_sound_2 is starting this frame...
                if instr_3_8_sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_8_sound_2.frameNStart = frameN  # exact frame index
                    instr_3_8_sound_2.tStart = t  # local t and not account for scr refresh
                    instr_3_8_sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_8_sound_2.started', tThisFlipGlobal)
                    # update status
                    instr_3_8_sound_2.status = STARTED
                    instr_3_8_sound_2.play(when=win)  # sync with win flip
                
                # if instr_3_8_sound_2 is stopping this frame...
                if instr_3_8_sound_2.status == STARTED:
                    if bool(False) or instr_3_8_sound_2.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_8_sound_2.tStop = t  # not accounting for scr refresh
                        instr_3_8_sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_8_sound_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_8_sound_2.stopped')
                        # update status
                        instr_3_8_sound_2.status = FINISHED
                        instr_3_8_sound_2.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_10,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_10.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_10.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_10" ---
            for thisComponent in instr_3_10.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_10
            instr_3_10.tStop = globalClock.getTime(format='float')
            instr_3_10.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_10.stopped', instr_3_10.tStop)
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            instr_loop.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                instr_loop.addData('key_resp_8.rt', key_resp_8.rt)
                instr_loop.addData('key_resp_8.duration', key_resp_8.duration)
            instr_3_8_sound_2.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_10" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_11" ---
            # create an object to store info about Routine instr_3_11
            instr_3_11 = data.Routine(
                name='instr_3_11',
                components=[text_27, text_28, text_29, text_30, text_31, text_41, text_32, key_resp_9, instr_3_9_sound_2],
            )
            instr_3_11.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_9
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            instr_3_9_sound_2.setSound('audio/instr_3_9.wav', hamming=True)
            instr_3_9_sound_2.setVolume(1.0, log=False)
            instr_3_9_sound_2.seek(0)
            # store start times for instr_3_11
            instr_3_11.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_11.tStart = globalClock.getTime(format='float')
            instr_3_11.status = STARTED
            thisExp.addData('instr_3_11.started', instr_3_11.tStart)
            instr_3_11.maxDuration = None
            # keep track of which components have finished
            instr_3_11Components = instr_3_11.components
            for thisComponent in instr_3_11.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_11" ---
            instr_3_11.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_27* updates
                
                # if text_27 is starting this frame...
                if text_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_27.frameNStart = frameN  # exact frame index
                    text_27.tStart = t  # local t and not account for scr refresh
                    text_27.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_27, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_27.started')
                    # update status
                    text_27.status = STARTED
                    text_27.setAutoDraw(True)
                
                # if text_27 is active this frame...
                if text_27.status == STARTED:
                    # update params
                    pass
                
                # *text_28* updates
                
                # if text_28 is starting this frame...
                if text_28.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_28.frameNStart = frameN  # exact frame index
                    text_28.tStart = t  # local t and not account for scr refresh
                    text_28.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_28, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_28.started')
                    # update status
                    text_28.status = STARTED
                    text_28.setAutoDraw(True)
                
                # if text_28 is active this frame...
                if text_28.status == STARTED:
                    # update params
                    pass
                
                # *text_29* updates
                
                # if text_29 is starting this frame...
                if text_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_29.frameNStart = frameN  # exact frame index
                    text_29.tStart = t  # local t and not account for scr refresh
                    text_29.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_29, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_29.started')
                    # update status
                    text_29.status = STARTED
                    text_29.setAutoDraw(True)
                
                # if text_29 is active this frame...
                if text_29.status == STARTED:
                    # update params
                    pass
                
                # *text_30* updates
                
                # if text_30 is starting this frame...
                if text_30.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_30.frameNStart = frameN  # exact frame index
                    text_30.tStart = t  # local t and not account for scr refresh
                    text_30.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_30, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_30.started')
                    # update status
                    text_30.status = STARTED
                    text_30.setAutoDraw(True)
                
                # if text_30 is active this frame...
                if text_30.status == STARTED:
                    # update params
                    pass
                
                # *text_31* updates
                
                # if text_31 is starting this frame...
                if text_31.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_31.frameNStart = frameN  # exact frame index
                    text_31.tStart = t  # local t and not account for scr refresh
                    text_31.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_31, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_31.started')
                    # update status
                    text_31.status = STARTED
                    text_31.setAutoDraw(True)
                
                # if text_31 is active this frame...
                if text_31.status == STARTED:
                    # update params
                    pass
                
                # *text_41* updates
                
                # if text_41 is starting this frame...
                if text_41.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_41.frameNStart = frameN  # exact frame index
                    text_41.tStart = t  # local t and not account for scr refresh
                    text_41.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_41, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_41.started')
                    # update status
                    text_41.status = STARTED
                    text_41.setAutoDraw(True)
                
                # if text_41 is active this frame...
                if text_41.status == STARTED:
                    # update params
                    pass
                
                # *text_32* updates
                
                # if text_32 is starting this frame...
                if text_32.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_32.frameNStart = frameN  # exact frame index
                    text_32.tStart = t  # local t and not account for scr refresh
                    text_32.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_32, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_32.started')
                    # update status
                    text_32.status = STARTED
                    text_32.setAutoDraw(True)
                
                # if text_32 is active this frame...
                if text_32.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_9_sound_2* updates
                
                # if instr_3_9_sound_2 is starting this frame...
                if instr_3_9_sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_9_sound_2.frameNStart = frameN  # exact frame index
                    instr_3_9_sound_2.tStart = t  # local t and not account for scr refresh
                    instr_3_9_sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_9_sound_2.started', tThisFlipGlobal)
                    # update status
                    instr_3_9_sound_2.status = STARTED
                    instr_3_9_sound_2.play(when=win)  # sync with win flip
                
                # if instr_3_9_sound_2 is stopping this frame...
                if instr_3_9_sound_2.status == STARTED:
                    if bool(False) or instr_3_9_sound_2.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_9_sound_2.tStop = t  # not accounting for scr refresh
                        instr_3_9_sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_9_sound_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_9_sound_2.stopped')
                        # update status
                        instr_3_9_sound_2.status = FINISHED
                        instr_3_9_sound_2.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_11,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_11.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_11.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_11" ---
            for thisComponent in instr_3_11.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_11
            instr_3_11.tStop = globalClock.getTime(format='float')
            instr_3_11.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_11.stopped', instr_3_11.tStop)
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            instr_loop.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                instr_loop.addData('key_resp_9.rt', key_resp_9.rt)
                instr_loop.addData('key_resp_9.duration', key_resp_9.duration)
            instr_3_9_sound_2.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_11" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_12" ---
            # create an object to store info about Routine instr_3_12
            instr_3_12 = data.Routine(
                name='instr_3_12',
                components=[text_33, text_34, text_35, key_resp_10, instr_3_6_sound_2],
            )
            instr_3_12.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_10
            key_resp_10.keys = []
            key_resp_10.rt = []
            _key_resp_10_allKeys = []
            instr_3_6_sound_2.setSound('audio/instr_3_2.wav', hamming=True)
            instr_3_6_sound_2.setVolume(1.0, log=False)
            instr_3_6_sound_2.seek(0)
            # store start times for instr_3_12
            instr_3_12.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_12.tStart = globalClock.getTime(format='float')
            instr_3_12.status = STARTED
            thisExp.addData('instr_3_12.started', instr_3_12.tStart)
            instr_3_12.maxDuration = None
            # keep track of which components have finished
            instr_3_12Components = instr_3_12.components
            for thisComponent in instr_3_12.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_12" ---
            instr_3_12.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_33* updates
                
                # if text_33 is starting this frame...
                if text_33.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_33.frameNStart = frameN  # exact frame index
                    text_33.tStart = t  # local t and not account for scr refresh
                    text_33.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_33, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_33.started')
                    # update status
                    text_33.status = STARTED
                    text_33.setAutoDraw(True)
                
                # if text_33 is active this frame...
                if text_33.status == STARTED:
                    # update params
                    pass
                
                # *text_34* updates
                
                # if text_34 is starting this frame...
                if text_34.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_34.frameNStart = frameN  # exact frame index
                    text_34.tStart = t  # local t and not account for scr refresh
                    text_34.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_34, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_34.started')
                    # update status
                    text_34.status = STARTED
                    text_34.setAutoDraw(True)
                
                # if text_34 is active this frame...
                if text_34.status == STARTED:
                    # update params
                    pass
                
                # *text_35* updates
                
                # if text_35 is starting this frame...
                if text_35.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_35.frameNStart = frameN  # exact frame index
                    text_35.tStart = t  # local t and not account for scr refresh
                    text_35.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_35, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_35.started')
                    # update status
                    text_35.status = STARTED
                    text_35.setAutoDraw(True)
                
                # if text_35 is active this frame...
                if text_35.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_10* updates
                waitOnFlip = False
                
                # if key_resp_10 is starting this frame...
                if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_10.frameNStart = frameN  # exact frame index
                    key_resp_10.tStart = t  # local t and not account for scr refresh
                    key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_10.started')
                    # update status
                    key_resp_10.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_10.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_10.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_10_allKeys.extend(theseKeys)
                    if len(_key_resp_10_allKeys):
                        key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                        key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                        key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_6_sound_2* updates
                
                # if instr_3_6_sound_2 is starting this frame...
                if instr_3_6_sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_6_sound_2.frameNStart = frameN  # exact frame index
                    instr_3_6_sound_2.tStart = t  # local t and not account for scr refresh
                    instr_3_6_sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_6_sound_2.started', tThisFlipGlobal)
                    # update status
                    instr_3_6_sound_2.status = STARTED
                    instr_3_6_sound_2.play(when=win)  # sync with win flip
                
                # if instr_3_6_sound_2 is stopping this frame...
                if instr_3_6_sound_2.status == STARTED:
                    if bool(False) or instr_3_6_sound_2.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_6_sound_2.tStop = t  # not accounting for scr refresh
                        instr_3_6_sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_6_sound_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_6_sound_2.stopped')
                        # update status
                        instr_3_6_sound_2.status = FINISHED
                        instr_3_6_sound_2.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_12,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_12.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_12.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_12" ---
            for thisComponent in instr_3_12.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_12
            instr_3_12.tStop = globalClock.getTime(format='float')
            instr_3_12.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_12.stopped', instr_3_12.tStop)
            # check responses
            if key_resp_10.keys in ['', [], None]:  # No response was made
                key_resp_10.keys = None
            instr_loop.addData('key_resp_10.keys',key_resp_10.keys)
            if key_resp_10.keys != None:  # we had a response
                instr_loop.addData('key_resp_10.rt', key_resp_10.rt)
                instr_loop.addData('key_resp_10.duration', key_resp_10.duration)
            instr_3_6_sound_2.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_12" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "instr_3_13" ---
            # create an object to store info about Routine instr_3_13
            instr_3_13 = data.Routine(
                name='instr_3_13',
                components=[text_36, text_37, text_38, text_39, text_40, text_42, text_43, text_44, key_resp_11, instr_3_9_sound_3],
            )
            instr_3_13.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp_11
            key_resp_11.keys = []
            key_resp_11.rt = []
            _key_resp_11_allKeys = []
            instr_3_9_sound_3.setSound('audio/instr_3_7.wav', hamming=True)
            instr_3_9_sound_3.setVolume(1.0, log=False)
            instr_3_9_sound_3.seek(0)
            # store start times for instr_3_13
            instr_3_13.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            instr_3_13.tStart = globalClock.getTime(format='float')
            instr_3_13.status = STARTED
            thisExp.addData('instr_3_13.started', instr_3_13.tStart)
            instr_3_13.maxDuration = None
            # keep track of which components have finished
            instr_3_13Components = instr_3_13.components
            for thisComponent in instr_3_13.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "instr_3_13" ---
            instr_3_13.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInstr_loop, 'status') and thisInstr_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_36* updates
                
                # if text_36 is starting this frame...
                if text_36.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_36.frameNStart = frameN  # exact frame index
                    text_36.tStart = t  # local t and not account for scr refresh
                    text_36.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_36, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_36.started')
                    # update status
                    text_36.status = STARTED
                    text_36.setAutoDraw(True)
                
                # if text_36 is active this frame...
                if text_36.status == STARTED:
                    # update params
                    pass
                
                # *text_37* updates
                
                # if text_37 is starting this frame...
                if text_37.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_37.frameNStart = frameN  # exact frame index
                    text_37.tStart = t  # local t and not account for scr refresh
                    text_37.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_37, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_37.started')
                    # update status
                    text_37.status = STARTED
                    text_37.setAutoDraw(True)
                
                # if text_37 is active this frame...
                if text_37.status == STARTED:
                    # update params
                    pass
                
                # *text_38* updates
                
                # if text_38 is starting this frame...
                if text_38.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_38.frameNStart = frameN  # exact frame index
                    text_38.tStart = t  # local t and not account for scr refresh
                    text_38.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_38, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_38.started')
                    # update status
                    text_38.status = STARTED
                    text_38.setAutoDraw(True)
                
                # if text_38 is active this frame...
                if text_38.status == STARTED:
                    # update params
                    pass
                
                # *text_39* updates
                
                # if text_39 is starting this frame...
                if text_39.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_39.frameNStart = frameN  # exact frame index
                    text_39.tStart = t  # local t and not account for scr refresh
                    text_39.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_39, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_39.started')
                    # update status
                    text_39.status = STARTED
                    text_39.setAutoDraw(True)
                
                # if text_39 is active this frame...
                if text_39.status == STARTED:
                    # update params
                    pass
                
                # *text_40* updates
                
                # if text_40 is starting this frame...
                if text_40.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_40.frameNStart = frameN  # exact frame index
                    text_40.tStart = t  # local t and not account for scr refresh
                    text_40.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_40, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_40.started')
                    # update status
                    text_40.status = STARTED
                    text_40.setAutoDraw(True)
                
                # if text_40 is active this frame...
                if text_40.status == STARTED:
                    # update params
                    pass
                
                # *text_42* updates
                
                # if text_42 is starting this frame...
                if text_42.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_42.frameNStart = frameN  # exact frame index
                    text_42.tStart = t  # local t and not account for scr refresh
                    text_42.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_42, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_42.started')
                    # update status
                    text_42.status = STARTED
                    text_42.setAutoDraw(True)
                
                # if text_42 is active this frame...
                if text_42.status == STARTED:
                    # update params
                    pass
                
                # *text_43* updates
                
                # if text_43 is starting this frame...
                if text_43.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_43.frameNStart = frameN  # exact frame index
                    text_43.tStart = t  # local t and not account for scr refresh
                    text_43.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_43, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_43.started')
                    # update status
                    text_43.status = STARTED
                    text_43.setAutoDraw(True)
                
                # if text_43 is active this frame...
                if text_43.status == STARTED:
                    # update params
                    pass
                
                # *text_44* updates
                
                # if text_44 is starting this frame...
                if text_44.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_44.frameNStart = frameN  # exact frame index
                    text_44.tStart = t  # local t and not account for scr refresh
                    text_44.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_44, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_44.started')
                    # update status
                    text_44.status = STARTED
                    text_44.setAutoDraw(True)
                
                # if text_44 is active this frame...
                if text_44.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_11* updates
                waitOnFlip = False
                
                # if key_resp_11 is starting this frame...
                if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_11.frameNStart = frameN  # exact frame index
                    key_resp_11.tStart = t  # local t and not account for scr refresh
                    key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_11.started')
                    # update status
                    key_resp_11.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_11.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_11.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_11_allKeys.extend(theseKeys)
                    if len(_key_resp_11_allKeys):
                        key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                        key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                        key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *instr_3_9_sound_3* updates
                
                # if instr_3_9_sound_3 is starting this frame...
                if instr_3_9_sound_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    instr_3_9_sound_3.frameNStart = frameN  # exact frame index
                    instr_3_9_sound_3.tStart = t  # local t and not account for scr refresh
                    instr_3_9_sound_3.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('instr_3_9_sound_3.started', tThisFlipGlobal)
                    # update status
                    instr_3_9_sound_3.status = STARTED
                    instr_3_9_sound_3.play(when=win)  # sync with win flip
                
                # if instr_3_9_sound_3 is stopping this frame...
                if instr_3_9_sound_3.status == STARTED:
                    if bool(False) or instr_3_9_sound_3.isFinished:
                        # keep track of stop time/frame for later
                        instr_3_9_sound_3.tStop = t  # not accounting for scr refresh
                        instr_3_9_sound_3.tStopRefresh = tThisFlipGlobal  # on global time
                        instr_3_9_sound_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'instr_3_9_sound_3.stopped')
                        # update status
                        instr_3_9_sound_3.status = FINISHED
                        instr_3_9_sound_3.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=instr_3_13,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    instr_3_13.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in instr_3_13.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instr_3_13" ---
            for thisComponent in instr_3_13.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for instr_3_13
            instr_3_13.tStop = globalClock.getTime(format='float')
            instr_3_13.tStopRefresh = tThisFlipGlobal
            thisExp.addData('instr_3_13.stopped', instr_3_13.tStop)
            # check responses
            if key_resp_11.keys in ['', [], None]:  # No response was made
                key_resp_11.keys = None
            instr_loop.addData('key_resp_11.keys',key_resp_11.keys)
            if key_resp_11.keys != None:  # we had a response
                instr_loop.addData('key_resp_11.rt', key_resp_11.rt)
                instr_loop.addData('key_resp_11.duration', key_resp_11.duration)
            instr_3_9_sound_3.pause()  # ensure sound has stopped at end of Routine
            # the Routine "instr_3_13" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisInstr_loop as finished
            if hasattr(thisInstr_loop, 'status'):
                thisInstr_loop.status = FINISHED
            # if awaiting a pause, pause now
            if instr_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                instr_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'instr_loop'
        instr_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "end_prac_judge" ---
        # create an object to store info about Routine end_prac_judge
        end_prac_judge = data.Routine(
            name='end_prac_judge',
            components=[into_prac, key_into_prac],
        )
        end_prac_judge.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_into_prac
        key_into_prac.keys = []
        key_into_prac.rt = []
        _key_into_prac_allKeys = []
        # store start times for end_prac_judge
        end_prac_judge.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        end_prac_judge.tStart = globalClock.getTime(format='float')
        end_prac_judge.status = STARTED
        thisExp.addData('end_prac_judge.started', end_prac_judge.tStart)
        end_prac_judge.maxDuration = None
        # keep track of which components have finished
        end_prac_judgeComponents = end_prac_judge.components
        for thisComponent in end_prac_judge.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "end_prac_judge" ---
        end_prac_judge.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_judgement, 'status') and thisPractice_judgement.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *into_prac* updates
            
            # if into_prac is starting this frame...
            if into_prac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                into_prac.frameNStart = frameN  # exact frame index
                into_prac.tStart = t  # local t and not account for scr refresh
                into_prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(into_prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'into_prac.started')
                # update status
                into_prac.status = STARTED
                into_prac.setAutoDraw(True)
            
            # if into_prac is active this frame...
            if into_prac.status == STARTED:
                # update params
                pass
            
            # *key_into_prac* updates
            waitOnFlip = False
            
            # if key_into_prac is starting this frame...
            if key_into_prac.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_into_prac.frameNStart = frameN  # exact frame index
                key_into_prac.tStart = t  # local t and not account for scr refresh
                key_into_prac.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_into_prac, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_into_prac.started')
                # update status
                key_into_prac.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_into_prac.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_into_prac.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_into_prac.status == STARTED and not waitOnFlip:
                theseKeys = key_into_prac.getKeys(keyList=['1','4'], ignoreKeys=["escape"], waitRelease=False)
                _key_into_prac_allKeys.extend(theseKeys)
                if len(_key_into_prac_allKeys):
                    key_into_prac.keys = _key_into_prac_allKeys[-1].name  # just the last key pressed
                    key_into_prac.rt = _key_into_prac_allKeys[-1].rt
                    key_into_prac.duration = _key_into_prac_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=end_prac_judge,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                end_prac_judge.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in end_prac_judge.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "end_prac_judge" ---
        for thisComponent in end_prac_judge.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for end_prac_judge
        end_prac_judge.tStop = globalClock.getTime(format='float')
        end_prac_judge.tStopRefresh = tThisFlipGlobal
        thisExp.addData('end_prac_judge.stopped', end_prac_judge.tStop)
        # Run 'End Routine' code from code_8
        if key_into_prac.keys=='4':
            practice_judgement.finished=True
        else:
            practice_judgement.finished=False
        # check responses
        if key_into_prac.keys in ['', [], None]:  # No response was made
            key_into_prac.keys = None
        practice_judgement.addData('key_into_prac.keys',key_into_prac.keys)
        if key_into_prac.keys != None:  # we had a response
            practice_judgement.addData('key_into_prac.rt', key_into_prac.rt)
            practice_judgement.addData('key_into_prac.duration', key_into_prac.duration)
        # the Routine "end_prac_judge" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPractice_judgement as finished
        if hasattr(thisPractice_judgement, 'status'):
            thisPractice_judgement.status = FINISHED
        # if awaiting a pause, pause now
        if practice_judgement.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_judgement.status = STARTED
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'practice_judgement'
    practice_judgement.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instr_4" ---
    # create an object to store info about Routine instr_4
    instr_4 = data.Routine(
        name='instr_4',
        components=[welpage_4, instr_key_resp_4, instr_4_sound],
    )
    instr_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_4
    instr_key_resp_4.keys = []
    instr_key_resp_4.rt = []
    _instr_key_resp_4_allKeys = []
    instr_4_sound.setSound('audio/instr_4.wav', hamming=True)
    instr_4_sound.setVolume(1.0, log=False)
    instr_4_sound.seek(0)
    # store start times for instr_4
    instr_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_4.tStart = globalClock.getTime(format='float')
    instr_4.status = STARTED
    thisExp.addData('instr_4.started', instr_4.tStart)
    instr_4.maxDuration = None
    # keep track of which components have finished
    instr_4Components = instr_4.components
    for thisComponent in instr_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_4" ---
    instr_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_4* updates
        
        # if welpage_4 is starting this frame...
        if welpage_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_4.frameNStart = frameN  # exact frame index
            welpage_4.tStart = t  # local t and not account for scr refresh
            welpage_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_4.started')
            # update status
            welpage_4.status = STARTED
            welpage_4.setAutoDraw(True)
        
        # if welpage_4 is active this frame...
        if welpage_4.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_4* updates
        waitOnFlip = False
        
        # if instr_key_resp_4 is starting this frame...
        if instr_key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_4.frameNStart = frameN  # exact frame index
            instr_key_resp_4.tStart = t  # local t and not account for scr refresh
            instr_key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_4.started')
            # update status
            instr_key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_4_allKeys.extend(theseKeys)
            if len(_instr_key_resp_4_allKeys):
                instr_key_resp_4.keys = _instr_key_resp_4_allKeys[-1].name  # just the last key pressed
                instr_key_resp_4.rt = _instr_key_resp_4_allKeys[-1].rt
                instr_key_resp_4.duration = _instr_key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_4_sound* updates
        
        # if instr_4_sound is starting this frame...
        if instr_4_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_4_sound.frameNStart = frameN  # exact frame index
            instr_4_sound.tStart = t  # local t and not account for scr refresh
            instr_4_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_4_sound.started', tThisFlipGlobal)
            # update status
            instr_4_sound.status = STARTED
            instr_4_sound.play(when=win)  # sync with win flip
        
        # if instr_4_sound is stopping this frame...
        if instr_4_sound.status == STARTED:
            if bool(False) or instr_4_sound.isFinished:
                # keep track of stop time/frame for later
                instr_4_sound.tStop = t  # not accounting for scr refresh
                instr_4_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_4_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_4_sound.stopped')
                # update status
                instr_4_sound.status = FINISHED
                instr_4_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_4,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_4" ---
    for thisComponent in instr_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_4
    instr_4.tStop = globalClock.getTime(format='float')
    instr_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_4.stopped', instr_4.tStop)
    # check responses
    if instr_key_resp_4.keys in ['', [], None]:  # No response was made
        instr_key_resp_4.keys = None
    thisExp.addData('instr_key_resp_4.keys',instr_key_resp_4.keys)
    if instr_key_resp_4.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_4.rt', instr_key_resp_4.rt)
        thisExp.addData('instr_key_resp_4.duration', instr_key_resp_4.duration)
    instr_4_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_4_3" ---
    # create an object to store info about Routine instr_4_3
    instr_4_3 = data.Routine(
        name='instr_4_3',
        components=[welpage_7, instr_key_resp_8, instr_4_sound_2],
    )
    instr_4_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_8
    instr_key_resp_8.keys = []
    instr_key_resp_8.rt = []
    _instr_key_resp_8_allKeys = []
    instr_4_sound_2.setSound('audio/instr_4_3.wav', hamming=True)
    instr_4_sound_2.setVolume(1.0, log=False)
    instr_4_sound_2.seek(0)
    # store start times for instr_4_3
    instr_4_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_4_3.tStart = globalClock.getTime(format='float')
    instr_4_3.status = STARTED
    thisExp.addData('instr_4_3.started', instr_4_3.tStart)
    instr_4_3.maxDuration = None
    # keep track of which components have finished
    instr_4_3Components = instr_4_3.components
    for thisComponent in instr_4_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_4_3" ---
    instr_4_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_7* updates
        
        # if welpage_7 is starting this frame...
        if welpage_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_7.frameNStart = frameN  # exact frame index
            welpage_7.tStart = t  # local t and not account for scr refresh
            welpage_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_7.started')
            # update status
            welpage_7.status = STARTED
            welpage_7.setAutoDraw(True)
        
        # if welpage_7 is active this frame...
        if welpage_7.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_8* updates
        waitOnFlip = False
        
        # if instr_key_resp_8 is starting this frame...
        if instr_key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_8.frameNStart = frameN  # exact frame index
            instr_key_resp_8.tStart = t  # local t and not account for scr refresh
            instr_key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_8.started')
            # update status
            instr_key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_8.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_8_allKeys.extend(theseKeys)
            if len(_instr_key_resp_8_allKeys):
                instr_key_resp_8.keys = _instr_key_resp_8_allKeys[-1].name  # just the last key pressed
                instr_key_resp_8.rt = _instr_key_resp_8_allKeys[-1].rt
                instr_key_resp_8.duration = _instr_key_resp_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_4_sound_2* updates
        
        # if instr_4_sound_2 is starting this frame...
        if instr_4_sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_4_sound_2.frameNStart = frameN  # exact frame index
            instr_4_sound_2.tStart = t  # local t and not account for scr refresh
            instr_4_sound_2.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_4_sound_2.started', tThisFlipGlobal)
            # update status
            instr_4_sound_2.status = STARTED
            instr_4_sound_2.play(when=win)  # sync with win flip
        
        # if instr_4_sound_2 is stopping this frame...
        if instr_4_sound_2.status == STARTED:
            if bool(False) or instr_4_sound_2.isFinished:
                # keep track of stop time/frame for later
                instr_4_sound_2.tStop = t  # not accounting for scr refresh
                instr_4_sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                instr_4_sound_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_4_sound_2.stopped')
                # update status
                instr_4_sound_2.status = FINISHED
                instr_4_sound_2.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_4_3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_4_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_4_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_4_3" ---
    for thisComponent in instr_4_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_4_3
    instr_4_3.tStop = globalClock.getTime(format='float')
    instr_4_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_4_3.stopped', instr_4_3.tStop)
    # check responses
    if instr_key_resp_8.keys in ['', [], None]:  # No response was made
        instr_key_resp_8.keys = None
    thisExp.addData('instr_key_resp_8.keys',instr_key_resp_8.keys)
    if instr_key_resp_8.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_8.rt', instr_key_resp_8.rt)
        thisExp.addData('instr_key_resp_8.duration', instr_key_resp_8.duration)
    instr_4_sound_2.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_4_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_4_1" ---
    # create an object to store info about Routine instr_4_1
    instr_4_1 = data.Routine(
        name='instr_4_1',
        components=[welpage_5, instr_key_resp_6, instr_4_1_sound],
    )
    instr_4_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_6
    instr_key_resp_6.keys = []
    instr_key_resp_6.rt = []
    _instr_key_resp_6_allKeys = []
    instr_4_1_sound.setSound('audio/instr_4_1.wav', hamming=True)
    instr_4_1_sound.setVolume(1.0, log=False)
    instr_4_1_sound.seek(0)
    # store start times for instr_4_1
    instr_4_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_4_1.tStart = globalClock.getTime(format='float')
    instr_4_1.status = STARTED
    thisExp.addData('instr_4_1.started', instr_4_1.tStart)
    instr_4_1.maxDuration = None
    # keep track of which components have finished
    instr_4_1Components = instr_4_1.components
    for thisComponent in instr_4_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_4_1" ---
    instr_4_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_5* updates
        
        # if welpage_5 is starting this frame...
        if welpage_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_5.frameNStart = frameN  # exact frame index
            welpage_5.tStart = t  # local t and not account for scr refresh
            welpage_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_5.started')
            # update status
            welpage_5.status = STARTED
            welpage_5.setAutoDraw(True)
        
        # if welpage_5 is active this frame...
        if welpage_5.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_6* updates
        waitOnFlip = False
        
        # if instr_key_resp_6 is starting this frame...
        if instr_key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_6.frameNStart = frameN  # exact frame index
            instr_key_resp_6.tStart = t  # local t and not account for scr refresh
            instr_key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_6.started')
            # update status
            instr_key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_6_allKeys.extend(theseKeys)
            if len(_instr_key_resp_6_allKeys):
                instr_key_resp_6.keys = _instr_key_resp_6_allKeys[-1].name  # just the last key pressed
                instr_key_resp_6.rt = _instr_key_resp_6_allKeys[-1].rt
                instr_key_resp_6.duration = _instr_key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_4_1_sound* updates
        
        # if instr_4_1_sound is starting this frame...
        if instr_4_1_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_4_1_sound.frameNStart = frameN  # exact frame index
            instr_4_1_sound.tStart = t  # local t and not account for scr refresh
            instr_4_1_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_4_1_sound.started', tThisFlipGlobal)
            # update status
            instr_4_1_sound.status = STARTED
            instr_4_1_sound.play(when=win)  # sync with win flip
        
        # if instr_4_1_sound is stopping this frame...
        if instr_4_1_sound.status == STARTED:
            if bool(False) or instr_4_1_sound.isFinished:
                # keep track of stop time/frame for later
                instr_4_1_sound.tStop = t  # not accounting for scr refresh
                instr_4_1_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_4_1_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_4_1_sound.stopped')
                # update status
                instr_4_1_sound.status = FINISHED
                instr_4_1_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_4_1,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_4_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_4_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_4_1" ---
    for thisComponent in instr_4_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_4_1
    instr_4_1.tStop = globalClock.getTime(format='float')
    instr_4_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_4_1.stopped', instr_4_1.tStop)
    # check responses
    if instr_key_resp_6.keys in ['', [], None]:  # No response was made
        instr_key_resp_6.keys = None
    thisExp.addData('instr_key_resp_6.keys',instr_key_resp_6.keys)
    if instr_key_resp_6.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_6.rt', instr_key_resp_6.rt)
        thisExp.addData('instr_key_resp_6.duration', instr_key_resp_6.duration)
    instr_4_1_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_4_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_4_2" ---
    # create an object to store info about Routine instr_4_2
    instr_4_2 = data.Routine(
        name='instr_4_2',
        components=[welpage_6, instr_key_resp_7, instr_4_2_sound],
    )
    instr_4_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_7
    instr_key_resp_7.keys = []
    instr_key_resp_7.rt = []
    _instr_key_resp_7_allKeys = []
    instr_4_2_sound.setSound('audio/instr_4_2.wav', hamming=True)
    instr_4_2_sound.setVolume(1.0, log=False)
    instr_4_2_sound.seek(0)
    # store start times for instr_4_2
    instr_4_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_4_2.tStart = globalClock.getTime(format='float')
    instr_4_2.status = STARTED
    thisExp.addData('instr_4_2.started', instr_4_2.tStart)
    instr_4_2.maxDuration = None
    # keep track of which components have finished
    instr_4_2Components = instr_4_2.components
    for thisComponent in instr_4_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_4_2" ---
    instr_4_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_6* updates
        
        # if welpage_6 is starting this frame...
        if welpage_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_6.frameNStart = frameN  # exact frame index
            welpage_6.tStart = t  # local t and not account for scr refresh
            welpage_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_6.started')
            # update status
            welpage_6.status = STARTED
            welpage_6.setAutoDraw(True)
        
        # if welpage_6 is active this frame...
        if welpage_6.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_7* updates
        waitOnFlip = False
        
        # if instr_key_resp_7 is starting this frame...
        if instr_key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_7.frameNStart = frameN  # exact frame index
            instr_key_resp_7.tStart = t  # local t and not account for scr refresh
            instr_key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_7.started')
            # update status
            instr_key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_7_allKeys.extend(theseKeys)
            if len(_instr_key_resp_7_allKeys):
                instr_key_resp_7.keys = _instr_key_resp_7_allKeys[-1].name  # just the last key pressed
                instr_key_resp_7.rt = _instr_key_resp_7_allKeys[-1].rt
                instr_key_resp_7.duration = _instr_key_resp_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_4_2_sound* updates
        
        # if instr_4_2_sound is starting this frame...
        if instr_4_2_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_4_2_sound.frameNStart = frameN  # exact frame index
            instr_4_2_sound.tStart = t  # local t and not account for scr refresh
            instr_4_2_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_4_2_sound.started', tThisFlipGlobal)
            # update status
            instr_4_2_sound.status = STARTED
            instr_4_2_sound.play(when=win)  # sync with win flip
        
        # if instr_4_2_sound is stopping this frame...
        if instr_4_2_sound.status == STARTED:
            if bool(False) or instr_4_2_sound.isFinished:
                # keep track of stop time/frame for later
                instr_4_2_sound.tStop = t  # not accounting for scr refresh
                instr_4_2_sound.tStopRefresh = tThisFlipGlobal  # on global time
                instr_4_2_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_4_2_sound.stopped')
                # update status
                instr_4_2_sound.status = FINISHED
                instr_4_2_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_4_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_4_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_4_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_4_2" ---
    for thisComponent in instr_4_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_4_2
    instr_4_2.tStop = globalClock.getTime(format='float')
    instr_4_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_4_2.stopped', instr_4_2.tStop)
    # check responses
    if instr_key_resp_7.keys in ['', [], None]:  # No response was made
        instr_key_resp_7.keys = None
    thisExp.addData('instr_key_resp_7.keys',instr_key_resp_7.keys)
    if instr_key_resp_7.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_7.rt', instr_key_resp_7.rt)
        thisExp.addData('instr_key_resp_7.duration', instr_key_resp_7.duration)
    instr_4_2_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_4_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    prac_block_loop = data.TrialHandler2(
        name='prac_block_loop',
        nReps=3.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(prac_block_loop)  # add the loop to the experiment
    thisPrac_block_loop = prac_block_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPrac_block_loop.rgb)
    if thisPrac_block_loop != None:
        for paramName in thisPrac_block_loop:
            globals()[paramName] = thisPrac_block_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPrac_block_loop in prac_block_loop:
        prac_block_loop.status = STARTED
        if hasattr(thisPrac_block_loop, 'status'):
            thisPrac_block_loop.status = STARTED
        currentLoop = prac_block_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_block_loop.rgb)
        if thisPrac_block_loop != None:
            for paramName in thisPrac_block_loop:
                globals()[paramName] = thisPrac_block_loop[paramName]
        
        # --- Prepare to start Routine "prac_rest" ---
        # create an object to store info about Routine prac_rest
        prac_rest = data.Routine(
            name='prac_rest',
            components=[prac_rest_instr, prac_rest_key_resp, prac_rest_sound],
        )
        prac_rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        if prac_block_loop.thisN == 0:
            prac_rest_text = '如果您准备好了，请按空格键开始练习。'
            prac_rest_audio = 'audio/prac_rest_instr_1.wav'
        else:
            prac_rest_text = '练习实验的正确率需达到60%才能进入正式实验，\n\n没关系，再来练习一次吧！\n\n\n请按空格键开始练习。'
            prac_rest_audio = 'audio/prac_rest_instr_2.wav'
        prac_rest_instr.setText(prac_rest_text)
        # create starting attributes for prac_rest_key_resp
        prac_rest_key_resp.keys = []
        prac_rest_key_resp.rt = []
        _prac_rest_key_resp_allKeys = []
        prac_rest_sound.setSound(prac_rest_audio, hamming=True)
        prac_rest_sound.setVolume(1.0, log=False)
        prac_rest_sound.seek(0)
        # store start times for prac_rest
        prac_rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_rest.tStart = globalClock.getTime(format='float')
        prac_rest.status = STARTED
        thisExp.addData('prac_rest.started', prac_rest.tStart)
        prac_rest.maxDuration = None
        # keep track of which components have finished
        prac_restComponents = prac_rest.components
        for thisComponent in prac_rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_rest" ---
        prac_rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_block_loop, 'status') and thisPrac_block_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_rest_instr* updates
            
            # if prac_rest_instr is starting this frame...
            if prac_rest_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_rest_instr.frameNStart = frameN  # exact frame index
                prac_rest_instr.tStart = t  # local t and not account for scr refresh
                prac_rest_instr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_rest_instr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_rest_instr.started')
                # update status
                prac_rest_instr.status = STARTED
                prac_rest_instr.setAutoDraw(True)
            
            # if prac_rest_instr is active this frame...
            if prac_rest_instr.status == STARTED:
                # update params
                pass
            
            # *prac_rest_key_resp* updates
            waitOnFlip = False
            
            # if prac_rest_key_resp is starting this frame...
            if prac_rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_rest_key_resp.frameNStart = frameN  # exact frame index
                prac_rest_key_resp.tStart = t  # local t and not account for scr refresh
                prac_rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_rest_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_rest_key_resp.started')
                # update status
                prac_rest_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_rest_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_rest_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_rest_key_resp_allKeys.extend(theseKeys)
                if len(_prac_rest_key_resp_allKeys):
                    prac_rest_key_resp.keys = _prac_rest_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_rest_key_resp.rt = _prac_rest_key_resp_allKeys[-1].rt
                    prac_rest_key_resp.duration = _prac_rest_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *prac_rest_sound* updates
            
            # if prac_rest_sound is starting this frame...
            if prac_rest_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_rest_sound.frameNStart = frameN  # exact frame index
                prac_rest_sound.tStart = t  # local t and not account for scr refresh
                prac_rest_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('prac_rest_sound.started', tThisFlipGlobal)
                # update status
                prac_rest_sound.status = STARTED
                prac_rest_sound.play(when=win)  # sync with win flip
            
            # if prac_rest_sound is stopping this frame...
            if prac_rest_sound.status == STARTED:
                if bool(False) or prac_rest_sound.isFinished:
                    # keep track of stop time/frame for later
                    prac_rest_sound.tStop = t  # not accounting for scr refresh
                    prac_rest_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_rest_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_rest_sound.stopped')
                    # update status
                    prac_rest_sound.status = FINISHED
                    prac_rest_sound.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_rest,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_rest" ---
        for thisComponent in prac_rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_rest
        prac_rest.tStop = globalClock.getTime(format='float')
        prac_rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_rest.stopped', prac_rest.tStop)
        # check responses
        if prac_rest_key_resp.keys in ['', [], None]:  # No response was made
            prac_rest_key_resp.keys = None
        prac_block_loop.addData('prac_rest_key_resp.keys',prac_rest_key_resp.keys)
        if prac_rest_key_resp.keys != None:  # we had a response
            prac_block_loop.addData('prac_rest_key_resp.rt', prac_rest_key_resp.rt)
            prac_block_loop.addData('prac_rest_key_resp.duration', prac_rest_key_resp.duration)
        prac_rest_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "prac_rest" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_fixation" ---
        # create an object to store info about Routine prac_fixation
        prac_fixation = data.Routine(
            name='prac_fixation',
            components=[prac_fix],
        )
        prac_fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for prac_fixation
        prac_fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_fixation.tStart = globalClock.getTime(format='float')
        prac_fixation.status = STARTED
        thisExp.addData('prac_fixation.started', prac_fixation.tStart)
        prac_fixation.maxDuration = None
        # keep track of which components have finished
        prac_fixationComponents = prac_fixation.components
        for thisComponent in prac_fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_fixation" ---
        prac_fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_block_loop, 'status') and thisPrac_block_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_fix* updates
            
            # if prac_fix is starting this frame...
            if prac_fix.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prac_fix.frameNStart = frameN  # exact frame index
                prac_fix.tStart = t  # local t and not account for scr refresh
                prac_fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_fix.started')
                # update status
                prac_fix.status = STARTED
                prac_fix.setAutoDraw(True)
            
            # if prac_fix is active this frame...
            if prac_fix.status == STARTED:
                # update params
                pass
            
            # if prac_fix is stopping this frame...
            if prac_fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_fix.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_fix.tStop = t  # not accounting for scr refresh
                    prac_fix.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_fix.stopped')
                    # update status
                    prac_fix.status = FINISHED
                    prac_fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_fixation,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_fixation" ---
        for thisComponent in prac_fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_fixation
        prac_fixation.tStop = globalClock.getTime(format='float')
        prac_fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_fixation.stopped', prac_fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_fixation.maxDurationReached:
            routineTimer.addTime(-prac_fixation.maxDuration)
        elif prac_fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # set up handler to look after randomisation of conditions etc
        prac_trials_loop = data.TrialHandler2(
            name='prac_trials_loop',
            nReps=5.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('trials_loop_zm.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(prac_trials_loop)  # add the loop to the experiment
        thisPrac_trials_loop = prac_trials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_trials_loop.rgb)
        if thisPrac_trials_loop != None:
            for paramName in thisPrac_trials_loop:
                globals()[paramName] = thisPrac_trials_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPrac_trials_loop in prac_trials_loop:
            prac_trials_loop.status = STARTED
            if hasattr(thisPrac_trials_loop, 'status'):
                thisPrac_trials_loop.status = STARTED
            currentLoop = prac_trials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPrac_trials_loop.rgb)
            if thisPrac_trials_loop != None:
                for paramName in thisPrac_trials_loop:
                    globals()[paramName] = thisPrac_trials_loop[paramName]
            
            # --- Prepare to start Routine "prac_exp" ---
            # create an object to store info about Routine prac_exp
            prac_exp = data.Routine(
                name='prac_exp',
                components=[prac_stimu, prac_blank, prac_key_resp],
            )
            prac_exp.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_2
            numlist = [0,1,2,3,4,5,6,7,8,9]
            shuffle(numlist)
            num = numlist.pop()
            count = count + 1
            if count > 2:
                if cond == 'same':
                    num = tempnum[1]
                    tempnum.pop(1)
                if cond == 'different':
                    while num == tempnum[1]:
                        num = numlist.pop()
                    tempnum.pop(1)
                    
            
            prac_stimu.setText(num)
            # create starting attributes for prac_key_resp
            prac_key_resp.keys = []
            prac_key_resp.rt = []
            _prac_key_resp_allKeys = []
            # store start times for prac_exp
            prac_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_exp.tStart = globalClock.getTime(format='float')
            prac_exp.status = STARTED
            thisExp.addData('prac_exp.started', prac_exp.tStart)
            prac_exp.maxDuration = None
            # keep track of which components have finished
            prac_expComponents = prac_exp.components
            for thisComponent in prac_exp.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_exp" ---
            prac_exp.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.26:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_trials_loop, 'status') and thisPrac_trials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_stimu* updates
                
                # if prac_stimu is starting this frame...
                if prac_stimu.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_stimu.frameNStart = frameN  # exact frame index
                    prac_stimu.tStart = t  # local t and not account for scr refresh
                    prac_stimu.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_stimu, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_stimu.started')
                    # update status
                    prac_stimu.status = STARTED
                    prac_stimu.setAutoDraw(True)
                
                # if prac_stimu is active this frame...
                if prac_stimu.status == STARTED:
                    # update params
                    pass
                
                # if prac_stimu is stopping this frame...
                if prac_stimu.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_stimu.tStartRefresh + 0.76-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_stimu.tStop = t  # not accounting for scr refresh
                        prac_stimu.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_stimu.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_stimu.stopped')
                        # update status
                        prac_stimu.status = FINISHED
                        prac_stimu.setAutoDraw(False)
                
                # *prac_blank* updates
                
                # if prac_blank is starting this frame...
                if prac_blank.status == NOT_STARTED and tThisFlip >= 0.76-frameTolerance:
                    # keep track of start time/frame for later
                    prac_blank.frameNStart = frameN  # exact frame index
                    prac_blank.tStart = t  # local t and not account for scr refresh
                    prac_blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_blank.started')
                    # update status
                    prac_blank.status = STARTED
                    prac_blank.setAutoDraw(True)
                
                # if prac_blank is active this frame...
                if prac_blank.status == STARTED:
                    # update params
                    pass
                
                # if prac_blank is stopping this frame...
                if prac_blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_blank.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_blank.tStop = t  # not accounting for scr refresh
                        prac_blank.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_blank.stopped')
                        # update status
                        prac_blank.status = FINISHED
                        prac_blank.setAutoDraw(False)
                
                # *prac_key_resp* updates
                waitOnFlip = False
                
                # if prac_key_resp is starting this frame...
                if prac_key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_key_resp.frameNStart = frameN  # exact frame index
                    prac_key_resp.tStart = t  # local t and not account for scr refresh
                    prac_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_key_resp.started')
                    # update status
                    prac_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if prac_key_resp is stopping this frame...
                if prac_key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_key_resp.tStartRefresh + 2.26-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_key_resp.tStop = t  # not accounting for scr refresh
                        prac_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_key_resp.stopped')
                        # update status
                        prac_key_resp.status = FINISHED
                        prac_key_resp.status = FINISHED
                if prac_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac_key_resp.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _prac_key_resp_allKeys.extend(theseKeys)
                    if len(_prac_key_resp_allKeys):
                        prac_key_resp.keys = _prac_key_resp_allKeys[-1].name  # just the last key pressed
                        prac_key_resp.rt = _prac_key_resp_allKeys[-1].rt
                        prac_key_resp.duration = _prac_key_resp_allKeys[-1].duration
                        # was this correct?
                        if (prac_key_resp.keys == str(correctAns)) or (prac_key_resp.keys == correctAns):
                            prac_key_resp.corr = 1
                        else:
                            prac_key_resp.corr = 0
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac_exp,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_exp.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_exp.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_exp" ---
            for thisComponent in prac_exp.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_exp
            prac_exp.tStop = globalClock.getTime(format='float')
            prac_exp.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_exp.stopped', prac_exp.tStop)
            # Run 'End Routine' code from code_2
            if count <= 2:
                prac_feedb_audio = 'audio/noresp.wav'
                    
            if count > 2:
                if not prac_key_resp.keys:
                    prac_feedb = '太慢!'
                    prac_feedb_audio = 'audio/tooslow.wav'
                    prac_key_resp.corr = 0
                elif prac_key_resp.corr == 1:
                    prac_feedb = '正确!'
                    prac_feedb_audio = 'audio/correct.wav'
                    #prac_feedbColor = 'green'
                    exp_rt.insert(0,prac_key_resp.rt)
                    #exp_rt.splice(0,0,prac_key_resp.rt)
                else:
                    prac_feedb = '错误!'
                    prac_feedb_audio = 'audio/wrong.wav'
                    #prac_feedbColor = 'red'
                exp_corr.insert(0,prac_key_resp.corr)
                #exp_corr.splice(0,0,prac_key_resp.corr)
            
            #offline py version
            tempnum.insert(0,num)
            #online js version
            #tempnum.splice(0,0,num)
            thisExp.addData('stimuli',num)
            thisExp.addData('exp_corr',exp_corr)
            thisExp.addData('prac_feedb_text',prac_feedb)
            # check responses
            if prac_key_resp.keys in ['', [], None]:  # No response was made
                prac_key_resp.keys = None
                # was no response the correct answer?!
                if str(correctAns).lower() == 'none':
                   prac_key_resp.corr = 1;  # correct non-response
                else:
                   prac_key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for prac_trials_loop (TrialHandler)
            prac_trials_loop.addData('prac_key_resp.keys',prac_key_resp.keys)
            prac_trials_loop.addData('prac_key_resp.corr', prac_key_resp.corr)
            if prac_key_resp.keys != None:  # we had a response
                prac_trials_loop.addData('prac_key_resp.rt', prac_key_resp.rt)
                prac_trials_loop.addData('prac_key_resp.duration', prac_key_resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_exp.maxDurationReached:
                routineTimer.addTime(-prac_exp.maxDuration)
            elif prac_exp.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.260000)
            
            # --- Prepare to start Routine "prac_trial_fb" ---
            # create an object to store info about Routine prac_trial_fb
            prac_trial_fb = data.Routine(
                name='prac_trial_fb',
                components=[prac_feedb_text, prac_feedb_sound],
            )
            prac_trial_fb.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            prac_feedb_text.setText(prac_feedb)
            prac_feedb_sound.setSound(prac_feedb_audio, secs=0.65, hamming=True)
            prac_feedb_sound.setVolume(1.0, log=False)
            prac_feedb_sound.seek(0)
            # store start times for prac_trial_fb
            prac_trial_fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac_trial_fb.tStart = globalClock.getTime(format='float')
            prac_trial_fb.status = STARTED
            thisExp.addData('prac_trial_fb.started', prac_trial_fb.tStart)
            prac_trial_fb.maxDuration = None
            # keep track of which components have finished
            prac_trial_fbComponents = prac_trial_fb.components
            for thisComponent in prac_trial_fb.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac_trial_fb" ---
            prac_trial_fb.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.65:
                # if trial has changed, end Routine now
                if hasattr(thisPrac_trials_loop, 'status') and thisPrac_trials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac_feedb_text* updates
                
                # if prac_feedb_text is starting this frame...
                if prac_feedb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_feedb_text.frameNStart = frameN  # exact frame index
                    prac_feedb_text.tStart = t  # local t and not account for scr refresh
                    prac_feedb_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac_feedb_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_feedb_text.started')
                    # update status
                    prac_feedb_text.status = STARTED
                    prac_feedb_text.setAutoDraw(True)
                
                # if prac_feedb_text is active this frame...
                if prac_feedb_text.status == STARTED:
                    # update params
                    pass
                
                # if prac_feedb_text is stopping this frame...
                if prac_feedb_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_feedb_text.tStartRefresh + 0.65-frameTolerance:
                        # keep track of stop time/frame for later
                        prac_feedb_text.tStop = t  # not accounting for scr refresh
                        prac_feedb_text.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_feedb_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_feedb_text.stopped')
                        # update status
                        prac_feedb_text.status = FINISHED
                        prac_feedb_text.setAutoDraw(False)
                
                # *prac_feedb_sound* updates
                
                # if prac_feedb_sound is starting this frame...
                if prac_feedb_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac_feedb_sound.frameNStart = frameN  # exact frame index
                    prac_feedb_sound.tStart = t  # local t and not account for scr refresh
                    prac_feedb_sound.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('prac_feedb_sound.started', tThisFlipGlobal)
                    # update status
                    prac_feedb_sound.status = STARTED
                    prac_feedb_sound.play(when=win)  # sync with win flip
                
                # if prac_feedb_sound is stopping this frame...
                if prac_feedb_sound.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac_feedb_sound.tStartRefresh + 0.65-frameTolerance or prac_feedb_sound.isFinished:
                        # keep track of stop time/frame for later
                        prac_feedb_sound.tStop = t  # not accounting for scr refresh
                        prac_feedb_sound.tStopRefresh = tThisFlipGlobal  # on global time
                        prac_feedb_sound.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac_feedb_sound.stopped')
                        # update status
                        prac_feedb_sound.status = FINISHED
                        prac_feedb_sound.stop()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac_trial_fb,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac_trial_fb.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac_trial_fb.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac_trial_fb" ---
            for thisComponent in prac_trial_fb.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac_trial_fb
            prac_trial_fb.tStop = globalClock.getTime(format='float')
            prac_trial_fb.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac_trial_fb.stopped', prac_trial_fb.tStop)
            prac_feedb_sound.pause()  # ensure sound has stopped at end of Routine
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac_trial_fb.maxDurationReached:
                routineTimer.addTime(-prac_trial_fb.maxDuration)
            elif prac_trial_fb.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.650000)
            # mark thisPrac_trials_loop as finished
            if hasattr(thisPrac_trials_loop, 'status'):
                thisPrac_trials_loop.status = FINISHED
            # if awaiting a pause, pause now
            if prac_trials_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                prac_trials_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 5.0 repeats of 'prac_trials_loop'
        prac_trials_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "prac_block_fb" ---
        # create an object to store info about Routine prac_block_fb
        prac_block_fb = data.Routine(
            name='prac_block_fb',
            components=[prac_block_feedb_text, prac_block_fb_key_resp],
        )
        prac_block_fb.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        exp_acc = (sum(exp_corr) / len(exp_corr) ) * 100;
        
        if len(exp_rt) == 0:
            block_feedb = '您的正确率为0。\n请确保明白实验规则再继续实验。\n\n\n请按空格键继续。';
        else:
            exp_meanRT = sum(exp_rt) / len(exp_rt);
            block_feedb = '您的正确率为%.2f%%\n您的平均反应时为%.2f秒\n\n\n请按空格键继续。'%(exp_acc,exp_meanRT);
        
        
        prac_block_feedb_text.setText(block_feedb)
        # create starting attributes for prac_block_fb_key_resp
        prac_block_fb_key_resp.keys = []
        prac_block_fb_key_resp.rt = []
        _prac_block_fb_key_resp_allKeys = []
        # store start times for prac_block_fb
        prac_block_fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_block_fb.tStart = globalClock.getTime(format='float')
        prac_block_fb.status = STARTED
        thisExp.addData('prac_block_fb.started', prac_block_fb.tStart)
        prac_block_fb.maxDuration = None
        # keep track of which components have finished
        prac_block_fbComponents = prac_block_fb.components
        for thisComponent in prac_block_fb.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_block_fb" ---
        prac_block_fb.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPrac_block_loop, 'status') and thisPrac_block_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_block_feedb_text* updates
            
            # if prac_block_feedb_text is starting this frame...
            if prac_block_feedb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_block_feedb_text.frameNStart = frameN  # exact frame index
                prac_block_feedb_text.tStart = t  # local t and not account for scr refresh
                prac_block_feedb_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_block_feedb_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_block_feedb_text.started')
                # update status
                prac_block_feedb_text.status = STARTED
                prac_block_feedb_text.setAutoDraw(True)
            
            # if prac_block_feedb_text is active this frame...
            if prac_block_feedb_text.status == STARTED:
                # update params
                pass
            
            # *prac_block_fb_key_resp* updates
            waitOnFlip = False
            
            # if prac_block_fb_key_resp is starting this frame...
            if prac_block_fb_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_block_fb_key_resp.frameNStart = frameN  # exact frame index
                prac_block_fb_key_resp.tStart = t  # local t and not account for scr refresh
                prac_block_fb_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_block_fb_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_block_fb_key_resp.started')
                # update status
                prac_block_fb_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(prac_block_fb_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(prac_block_fb_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if prac_block_fb_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = prac_block_fb_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _prac_block_fb_key_resp_allKeys.extend(theseKeys)
                if len(_prac_block_fb_key_resp_allKeys):
                    prac_block_fb_key_resp.keys = _prac_block_fb_key_resp_allKeys[-1].name  # just the last key pressed
                    prac_block_fb_key_resp.rt = _prac_block_fb_key_resp_allKeys[-1].rt
                    prac_block_fb_key_resp.duration = _prac_block_fb_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=prac_block_fb,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_block_fb.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_block_fb.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_block_fb" ---
        for thisComponent in prac_block_fb.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_block_fb
        prac_block_fb.tStop = globalClock.getTime(format='float')
        prac_block_fb.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_block_fb.stopped', prac_block_fb.tStop)
        # Run 'End Routine' code from code_5
        count = 0
        exp_corr = []
        exp_rt = []
        prac_feedb = ''
        
        if exp_acc >= 60:
            prac_block_loop.finished = True
            
        # check responses
        if prac_block_fb_key_resp.keys in ['', [], None]:  # No response was made
            prac_block_fb_key_resp.keys = None
        prac_block_loop.addData('prac_block_fb_key_resp.keys',prac_block_fb_key_resp.keys)
        if prac_block_fb_key_resp.keys != None:  # we had a response
            prac_block_loop.addData('prac_block_fb_key_resp.rt', prac_block_fb_key_resp.rt)
            prac_block_loop.addData('prac_block_fb_key_resp.duration', prac_block_fb_key_resp.duration)
        # the Routine "prac_block_fb" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPrac_block_loop as finished
        if hasattr(thisPrac_block_loop, 'status'):
            thisPrac_block_loop.status = FINISHED
        # if awaiting a pause, pause now
        if prac_block_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            prac_block_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'prac_block_loop'
    prac_block_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instr_5" ---
    # create an object to store info about Routine instr_5
    instr_5 = data.Routine(
        name='instr_5',
        components=[welpage_exp, instr_key_resp_5, instr_sound_5],
    )
    instr_5.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key_resp_5
    instr_key_resp_5.keys = []
    instr_key_resp_5.rt = []
    _instr_key_resp_5_allKeys = []
    instr_sound_5.setSound('audio/instr_5_new.wav', hamming=True)
    instr_sound_5.setVolume(1.0, log=False)
    instr_sound_5.seek(0)
    # store start times for instr_5
    instr_5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_5.tStart = globalClock.getTime(format='float')
    instr_5.status = STARTED
    thisExp.addData('instr_5.started', instr_5.tStart)
    instr_5.maxDuration = None
    # keep track of which components have finished
    instr_5Components = instr_5.components
    for thisComponent in instr_5.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instr_5" ---
    instr_5.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welpage_exp* updates
        
        # if welpage_exp is starting this frame...
        if welpage_exp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welpage_exp.frameNStart = frameN  # exact frame index
            welpage_exp.tStart = t  # local t and not account for scr refresh
            welpage_exp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welpage_exp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welpage_exp.started')
            # update status
            welpage_exp.status = STARTED
            welpage_exp.setAutoDraw(True)
        
        # if welpage_exp is active this frame...
        if welpage_exp.status == STARTED:
            # update params
            pass
        
        # *instr_key_resp_5* updates
        waitOnFlip = False
        
        # if instr_key_resp_5 is starting this frame...
        if instr_key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key_resp_5.frameNStart = frameN  # exact frame index
            instr_key_resp_5.tStart = t  # local t and not account for scr refresh
            instr_key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key_resp_5.started')
            # update status
            instr_key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = instr_key_resp_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_resp_5_allKeys.extend(theseKeys)
            if len(_instr_key_resp_5_allKeys):
                instr_key_resp_5.keys = _instr_key_resp_5_allKeys[-1].name  # just the last key pressed
                instr_key_resp_5.rt = _instr_key_resp_5_allKeys[-1].rt
                instr_key_resp_5.duration = _instr_key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *instr_sound_5* updates
        
        # if instr_sound_5 is starting this frame...
        if instr_sound_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_sound_5.frameNStart = frameN  # exact frame index
            instr_sound_5.tStart = t  # local t and not account for scr refresh
            instr_sound_5.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('instr_sound_5.started', tThisFlipGlobal)
            # update status
            instr_sound_5.status = STARTED
            instr_sound_5.play(when=win)  # sync with win flip
        
        # if instr_sound_5 is stopping this frame...
        if instr_sound_5.status == STARTED:
            if bool(False) or instr_sound_5.isFinished:
                # keep track of stop time/frame for later
                instr_sound_5.tStop = t  # not accounting for scr refresh
                instr_sound_5.tStopRefresh = tThisFlipGlobal  # on global time
                instr_sound_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instr_sound_5.stopped')
                # update status
                instr_sound_5.status = FINISHED
                instr_sound_5.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instr_5,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instr_5.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_5.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_5" ---
    for thisComponent in instr_5.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_5
    instr_5.tStop = globalClock.getTime(format='float')
    instr_5.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_5.stopped', instr_5.tStop)
    # check responses
    if instr_key_resp_5.keys in ['', [], None]:  # No response was made
        instr_key_resp_5.keys = None
    thisExp.addData('instr_key_resp_5.keys',instr_key_resp_5.keys)
    if instr_key_resp_5.keys != None:  # we had a response
        thisExp.addData('instr_key_resp_5.rt', instr_key_resp_5.rt)
        thisExp.addData('instr_key_resp_5.duration', instr_key_resp_5.duration)
    instr_sound_5.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "instr_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    block_loop = data.TrialHandler2(
        name='block_loop',
        nReps=n_blocks, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(block_loop)  # add the loop to the experiment
    thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop:
            globals()[paramName] = thisBlock_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock_loop in block_loop:
        block_loop.status = STARTED
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = STARTED
        currentLoop = block_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
        if thisBlock_loop != None:
            for paramName in thisBlock_loop:
                globals()[paramName] = thisBlock_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[fix],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix* updates
            
            # if fix is starting this frame...
            if fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix.frameNStart = frameN  # exact frame index
                fix.tStart = t  # local t and not account for scr refresh
                fix.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix.started')
                # update status
                fix.status = STARTED
                fix.setAutoDraw(True)
            
            # if fix is active this frame...
            if fix.status == STARTED:
                # update params
                pass
            
            # if fix is stopping this frame...
            if fix.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix.tStop = t  # not accounting for scr refresh
                    fix.tStopRefresh = tThisFlipGlobal  # on global time
                    fix.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix.stopped')
                    # update status
                    fix.status = FINISHED
                    fix.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # set up handler to look after randomisation of conditions etc
        trials_loop = data.TrialHandler2(
            name='trials_loop',
            nReps=5.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('trials_loop_zm.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(trials_loop)  # add the loop to the experiment
        thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_loop in trials_loop:
            trials_loop.status = STARTED
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = STARTED
            currentLoop = trials_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
            if thisTrials_loop != None:
                for paramName in thisTrials_loop:
                    globals()[paramName] = thisTrials_loop[paramName]
            
            # --- Prepare to start Routine "exp" ---
            # create an object to store info about Routine exp
            exp = data.Routine(
                name='exp',
                components=[stimu, blank, key_resp],
            )
            exp.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            numlist = [0,1,2,3,4,5,6,7,8,9]
            shuffle(numlist)
            num = numlist.pop()
            count = count + 1
            if count > 2:
                if cond == 'same':
                    num = tempnum[1]
                    tempnum.pop(1)
                if cond == 'different':
                    while num == tempnum[1]:
                        num = numlist.pop()
                    tempnum.pop(1)
                    
            
            stimu.setText(num)
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # store start times for exp
            exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            exp.tStart = globalClock.getTime(format='float')
            exp.status = STARTED
            thisExp.addData('exp.started', exp.tStart)
            exp.maxDuration = None
            # keep track of which components have finished
            expComponents = exp.components
            for thisComponent in exp.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "exp" ---
            exp.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.26:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_loop, 'status') and thisTrials_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *stimu* updates
                
                # if stimu is starting this frame...
                if stimu.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    stimu.frameNStart = frameN  # exact frame index
                    stimu.tStart = t  # local t and not account for scr refresh
                    stimu.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(stimu, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimu.started')
                    # update status
                    stimu.status = STARTED
                    stimu.setAutoDraw(True)
                
                # if stimu is active this frame...
                if stimu.status == STARTED:
                    # update params
                    pass
                
                # if stimu is stopping this frame...
                if stimu.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > stimu.tStartRefresh + 0.76-frameTolerance:
                        # keep track of stop time/frame for later
                        stimu.tStop = t  # not accounting for scr refresh
                        stimu.tStopRefresh = tThisFlipGlobal  # on global time
                        stimu.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'stimu.stopped')
                        # update status
                        stimu.status = FINISHED
                        stimu.setAutoDraw(False)
                
                # *blank* updates
                
                # if blank is starting this frame...
                if blank.status == NOT_STARTED and tThisFlip >= 0.76-frameTolerance:
                    # keep track of start time/frame for later
                    blank.frameNStart = frameN  # exact frame index
                    blank.tStart = t  # local t and not account for scr refresh
                    blank.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(blank, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank.started')
                    # update status
                    blank.status = STARTED
                    blank.setAutoDraw(True)
                
                # if blank is active this frame...
                if blank.status == STARTED:
                    # update params
                    pass
                
                # if blank is stopping this frame...
                if blank.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > blank.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        blank.tStop = t  # not accounting for scr refresh
                        blank.tStopRefresh = tThisFlipGlobal  # on global time
                        blank.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'blank.stopped')
                        # update status
                        blank.status = FINISHED
                        blank.setAutoDraw(False)
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp is stopping this frame...
                if key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp.tStartRefresh + 2.26-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        # update status
                        key_resp.status = FINISHED
                        key_resp.status = FINISHED
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['z','m'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # was this correct?
                        if (key_resp.keys == str(correctAns)) or (key_resp.keys == correctAns):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=exp,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    exp.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in exp.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "exp" ---
            for thisComponent in exp.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for exp
            exp.tStop = globalClock.getTime(format='float')
            exp.tStopRefresh = tThisFlipGlobal
            thisExp.addData('exp.stopped', exp.tStop)
            # Run 'End Routine' code from code
            if count > 2:
                if not key_resp.keys:
                    prac_feedb = '太慢!'
                    key_resp.corr = 0
                elif key_resp.corr == 1:
                    prac_feedb = '正确!'
                    prac_feedbColor = 'green'
                    exp_rt.insert(0,key_resp.rt)
                else:
                    prac_feedb = '错误!'
                    prac_feedbColor = 'red'
                exp_corr.insert(0,key_resp.corr)
            
            tempnum.insert(0,num)
            thisExp.addData('stimuli',num)
            thisExp.addData('exp_corr',exp_corr)
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correctAns).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_loop (TrialHandler)
            trials_loop.addData('key_resp.keys',key_resp.keys)
            trials_loop.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                trials_loop.addData('key_resp.rt', key_resp.rt)
                trials_loop.addData('key_resp.duration', key_resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if exp.maxDurationReached:
                routineTimer.addTime(-exp.maxDuration)
            elif exp.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.260000)
            # mark thisTrials_loop as finished
            if hasattr(thisTrials_loop, 'status'):
                thisTrials_loop.status = FINISHED
            # if awaiting a pause, pause now
            if trials_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 5.0 repeats of 'trials_loop'
        trials_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "block_fb" ---
        # create an object to store info about Routine block_fb
        block_fb = data.Routine(
            name='block_fb',
            components=[block_feedb_text, block_fb_key_resp],
        )
        block_fb.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        exp_acc = (sum(exp_corr) / len(exp_corr)) * 100
        blockn = n_blocks - (block_loop.thisN+1)
        print(blockn)
        if len(exp_rt) == 0:
            block_feedb = '您的正确率为0。\n请确保明白实验规则再继续实验。\n\n\n请按空格键继续。'
        else:
            exp_meanRT = sum(exp_rt)/len(exp_rt)
            block_feedb = '您的正确率为%.2f%%\n您的平均反应时为%.2f秒\n当前剩余%d组实验\n\n\n请按空格键继续。'%(exp_acc,exp_meanRT,blockn)
        block_feedb_text.setText(block_feedb)
        # create starting attributes for block_fb_key_resp
        block_fb_key_resp.keys = []
        block_fb_key_resp.rt = []
        _block_fb_key_resp_allKeys = []
        # store start times for block_fb
        block_fb.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_fb.tStart = globalClock.getTime(format='float')
        block_fb.status = STARTED
        thisExp.addData('block_fb.started', block_fb.tStart)
        block_fb.maxDuration = None
        # keep track of which components have finished
        block_fbComponents = block_fb.components
        for thisComponent in block_fb.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "block_fb" ---
        block_fb.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_feedb_text* updates
            
            # if block_feedb_text is starting this frame...
            if block_feedb_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_feedb_text.frameNStart = frameN  # exact frame index
                block_feedb_text.tStart = t  # local t and not account for scr refresh
                block_feedb_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_feedb_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_feedb_text.started')
                # update status
                block_feedb_text.status = STARTED
                block_feedb_text.setAutoDraw(True)
            
            # if block_feedb_text is active this frame...
            if block_feedb_text.status == STARTED:
                # update params
                pass
            
            # *block_fb_key_resp* updates
            waitOnFlip = False
            
            # if block_fb_key_resp is starting this frame...
            if block_fb_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_fb_key_resp.frameNStart = frameN  # exact frame index
                block_fb_key_resp.tStart = t  # local t and not account for scr refresh
                block_fb_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_fb_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_fb_key_resp.started')
                # update status
                block_fb_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(block_fb_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(block_fb_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if block_fb_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = block_fb_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _block_fb_key_resp_allKeys.extend(theseKeys)
                if len(_block_fb_key_resp_allKeys):
                    block_fb_key_resp.keys = _block_fb_key_resp_allKeys[-1].name  # just the last key pressed
                    block_fb_key_resp.rt = _block_fb_key_resp_allKeys[-1].rt
                    block_fb_key_resp.duration = _block_fb_key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=block_fb,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                block_fb.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_fb.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_fb" ---
        for thisComponent in block_fb.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_fb
        block_fb.tStop = globalClock.getTime(format='float')
        block_fb.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_fb.stopped', block_fb.tStop)
        # Run 'End Routine' code from code_3
        count = 0
        exp_corr = []
        exp_rt = []
        # check responses
        if block_fb_key_resp.keys in ['', [], None]:  # No response was made
            block_fb_key_resp.keys = None
        block_loop.addData('block_fb_key_resp.keys',block_fb_key_resp.keys)
        if block_fb_key_resp.keys != None:  # we had a response
            block_loop.addData('block_fb_key_resp.rt', block_fb_key_resp.rt)
            block_loop.addData('block_fb_key_resp.duration', block_fb_key_resp.duration)
        # the Routine "block_fb" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisBlock_loop as finished
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = FINISHED
        # if awaiting a pause, pause now
        if block_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            block_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed n_blocks repeats of 'block_loop'
    block_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[endpage],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endpage* updates
        
        # if endpage is starting this frame...
        if endpage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endpage.frameNStart = frameN  # exact frame index
            endpage.tStart = t  # local t and not account for scr refresh
            endpage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endpage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endpage.started')
            # update status
            endpage.status = STARTED
            endpage.setAutoDraw(True)
        
        # if endpage is active this frame...
        if endpage.status == STARTED:
            # update params
            pass
        
        # if endpage is stopping this frame...
        if endpage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > endpage.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                endpage.tStop = t  # not accounting for scr refresh
                endpage.tStopRefresh = tThisFlipGlobal  # on global time
                endpage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'endpage.stopped')
                # update status
                endpage.status = FINISHED
                endpage.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thanks,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks.maxDurationReached:
        routineTimer.addTime(-thanks.maxDuration)
    elif thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
