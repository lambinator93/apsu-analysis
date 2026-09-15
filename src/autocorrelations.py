import numpy as np

"""""""""
--- G2 Autocorrelation Function
"""""""""
def g2calc(Img, mask,dpl=4):
    """""""""
    G2CALC: Function to calculate pre-initialized tensors 'G2, IP & IF' in
    a multi-tau G2 calculation scheme from a stack of 'nframes' 
    pre-processed (dark, flatfield, blemish, and lld) images 'Img.' 
    In addition, the pre-average image stacks 'LImg' and the counters 'GC' 
    and 'LC' are updates

    This time, the image stack is of this order 'Img(nframs, rows, cols)'

    """""""""
    
    #dpl = 4
    
    nframes, rows, cols = Img.shape
    
    # --- Get delay and maxlevel
    maxLevel = int(getdelayinfo(nframes,dpl)[1])
    
    # --- Vector for integrated intensity
    sumI = np.zeros((1,nframes))
    
    # --- Image stacks for all Levels
    LImg = np.zeros((int(2*dpl),maxLevel,rows,cols),dtype=np.single) # initialize image arrays (ROI size) for all levels
    LC = np.zeros((1,maxLevel)) # initialize level counter
    
    # --- G2 tensor & IF & IP + counter
    G2 = np.zeros(((maxLevel+1)*dpl-1,rows,cols),dtype=np.single) # initialize the G2 tensor (ROI size)
    IF = np.zeros(((maxLevel+1)*dpl-1,rows,cols),dtype=np.single) # initialize the IF (intensities future) tensor (ROI size)
    IP = np.zeros(((maxLevel+1)*dpl-1,rows,cols),dtype=np.single) # initialize the IP (intensities past) tensor (ROI size)
    GC = np.zeros((1, (maxLevel+1)*dpl-1)) # counter for the G2 sums
    
    # --- Start of loop over all frames
    for i in range(1,nframes+1,1):
        # --- Level 1: a) update image stacks
        iFrame = Img[i-1,:,:]
        ImgMask = iFrame[mask == 1] # unmasked pixels on the ROI
        sumI[0,i-1] = np.sum(ImgMask)
        # --- 
        LImg[((i-1)%(2*dpl)),0,:,:] = Img[i-1,:,:] # Replace the oldest image in the level 1 stack with the new one
        LC[0,0] += 1 # Increment counter for first level
        
        for l in range(1,2*dpl,1): # loop over all level 1 delays
            if i >= (l+1):
                #print('l: {}'.format(l))
                # latest image times an 'l' times older image
                G2[l-1,:,:] += LImg[((i-1)%(2*dpl)),0,:,:]*LImg[((i-1-l)%(2*dpl)),0,:,:] 
                IF[l-1,:,:] += LImg[((i-1)%(2*dpl)),0,:,:] # add to IF tensor
                IP[l-1,:,:] += LImg[((i-1-l)%(2*dpl)),0,:,:] # add to IP tensor
                GC[0,l-1] += 1 # increase G2 counter
            
        # --- Higher Levels
        for k in range(2,maxLevel+1,1):                          # loop over all higher levels
            # --- Higher Levels: a) update image stacks
            if (i%(2**(k-1)) == 0):
                # average (k-1) level images into level k
                LImg[int(LC[0,k-1]%(2*dpl)),k-1,:,:] = 0.5*(LImg[int((LC[0,k-2]-2)%(2*dpl)),k-2,:,:]+LImg[int((LC[0,k-2]-1)%(2*dpl)),k-2,:,:]) 
                LC[0,k-1] += 1        # increase level k counter
                
            # --- Higher Levels: b) perform multi-tau G2 calculation
            for l in range(1,dpl+1,1):    # loop over all delays of level k
                n = k*dpl + l - 1   # delay number
                g2n = dpl*2**(k-1)+l*2**(k-1)   # 1st change to add to the delay n of g2
                
                #print('n: {}'.format(n))
                if (i%2**(k-1)) == g2n%2**(k-1) and i>=g2n:
                    # latest image times an 'l' times older image
                    G2[n-1,:,:] += LImg[int((GC[0,n-1]+dpl+l-1)%(2*dpl)),k-1,:,:]*LImg[int(GC[0,n-1]%(2*dpl)),k-1,:,:]   
                    IF[n-1,:,:] += LImg[int((GC[0,n-1]+dpl+l-1)%(2*dpl)),k-1,:,:]   # add to IF tensor
                    IP[n-1,:,:] += LImg[int(GC[0,n-1]%(2*dpl)),k-1,:,:]   # add to IP tensor
                    GC[0,n-1] += 1   # increase G2 counter
                    
    # --- Normalize multi-tau tensors
    for k in range(GC.shape[1]):
        if (GC[0,k] != 0):
            G2[k,:,:] = G2[k,:,:] / GC[0,k]
            IF[k,:,:] = IF[k,:,:] / GC[0,k]
            IP[k,:,:] = IP[k,:,:] / GC[0,k]

    # --- Return variables
    return sumI, G2, IF, IP, GC


"""""""""
--- Two-Time Correlation Function
"""""""""

def TwoTime(Img):
    """
    TWOTIME: Function to calculate the two-time correlation matrix.
    INPUT: Img(nframes, rows, cols) - Stack of images.
    OUTPUT: Correlation Matrix C(nframes, nframes)
    """
    
    n2tframes = Img.shape[0]  # Number of time frames
    N_pixels = Img.shape[1] * Img.shape[2]  # Total pixels per frame
    C = np.zeros((n2tframes, n2tframes), dtype=np.single)

    # Compute mean intensity for each time frame
    mean_intensity = np.mean(Img, axis=(1, 2))  # Shape (n2tframes,)

    # Compute correlation matrix
    for k in range(n2tframes):
        for l in range(k, n2tframes):
            # Compute intensity product averaged over pixels
            numerator = np.sum(Img[k, :, :] * Img[l, :, :]) / N_pixels
            denominator = mean_intensity[k] * mean_intensity[l]  # Product of means
            C[k, l] = numerator / denominator  # Apply normalization

    # Fill upper triangle (C is symmetric)
    for k in range(n2tframes):
        C[k+1:n2tframes, k] = C[k, k+1:n2tframes]

    return C

def twotime(Img):
    """""""""
    TWOTIME: Function to calculate the two-time correlation matrix. Seems somewhat slow

    INPUT: This time, the image stack is of this order 'Img(nframes, rows, cols).' 
    
    OUTPUT: Correlation Matrix C(nframes,nframes)
    """""""""
    
    # --- Reshapes array to 
    #Img = np.reshape(Img,(Img.shape[0],Img.shape[1]*Img.shape[2]))
    
    n2tframes = Img.shape[0]
    C = np.zeros((n2tframes, n2tframes), dtype=np.single)
    
    # --- Some pre-calculations
    #Iaqt = np.zeros((n2tframes,1,1), dtype=np.single)
    
    """""
    for k in range(n2tframes):
        Iaqt[k,0,0] = np.mean(Img[k,:,:])
        #Mark's Version
        Img[k,:,:] = (Img[k,:,:]-Iaqt[k,0,0])/Iaqt[k,0,0]
    """""
    New_Img = np.zeros((Img.shape))
    for k in range(n2tframes):
        Iaqt = np.mean(Img[k,:,:])
        #Mark's Version
        New_Img[k,:,:] = (Img[k,:,:]-Iaqt)/Iaqt
        
    # --- Main loop
    for k in range(n2tframes):
        for l in range(n2tframes):
            if l < k:
                # Modified Mark version: summing (normalize later)
                C[k,l] = np.sum(New_Img[k,:,:]*New_Img[l,:,:])
                                
    # --- Fill upper corner
    for k in range(n2tframes):
        C[k,k+1:n2tframes] = C[k+1:n2tframes,k]
    
    # --- Correct Diagonal
    for k in range(n2tframes):
        if k != 0 and k != (n2tframes-1):
            #Calculate the diagonal as average
            C[k,k] = (1/4)*(C[k-1,k]+C[k+1,k]+C[k,k-1]+C[k,k+1])
        elif k == 0:
            #Calculate the diagonal as average
            C[k,k] = (1/2)*(C[k+1,k]+C[k,k-1])
        elif k == n2tframes-1:
            #Calculate the diagonal as average
            C[k,k] = (1/2)*(C[k-1,k]+C[k,k-1])
    
    # --- Normalize
    C/=Img.shape[1]
    
    return C

def diff_matrix(Img):
    """""""""
    TWOTIME: Function to calculate the two-time correlation matrix. Seems somewhat slow

    INPUT: This time, the image stack is of this order 'Img(nframes, rows, cols).' 
    
    OUTPUT: Difference Matrix D(nframes,nframes)
    """""""""
    
    # --- Reshapes array to 
    #Img = np.reshape(Img,(Img.shape[0],Img.shape[1]*Img.shape[2]))
    
    n2tframes = Img.shape[0]
    D = np.zeros((n2tframes, n2tframes), dtype=np.single)
    
    # --- Some pre-calculations
    #Iaqt = np.zeros((n2tframes,1,1), dtype=np.single)
    
    """""
    for k in range(n2tframes):
        Iaqt[k,0,0] = np.mean(Img[k,:,:])
        #Mark's Version
        Img[k,:,:] = (Img[k,:,:]-Iaqt[k,0,0])/Iaqt[k,0,0]
    """""
    New_Img = np.zeros((Img.shape))
    for k in range(n2tframes):
        Iaqt = np.mean(Img[k,:,:])
        #Mark's Version
        New_Img[k,:,:] = (Img[k,:,:])#-Iaqt)/Iaqt
        
    # --- Main loop
    for k in range(n2tframes):
        for l in range(n2tframes):
            if l < k:
                # Modified Mark version: summing (normalize later)
                D[k,l] = np.mean(np.abs(New_Img[k,:,:]-New_Img[l,:,:]))
                                
    # --- Fill upper corner
    for k in range(n2tframes):
        D[k,k+1:n2tframes] = D[k+1:n2tframes,k]
    
    # --- Correct Diagonal
    for k in range(n2tframes):
        if k != 0 and k != (n2tframes-1):
            #Calculate the diagonal as average
            D[k,k] = (1/4)*(D[k-1,k]+D[k+1,k]+D[k,k-1]+D[k,k+1])
        elif k == 0:
            #Calculate the diagonal as average
            D[k,k] = (1/2)*(D[k+1,k]+D[k,k-1])
        elif k == n2tframes:
            #Calculate the diagonal as average
            D[k,k] = (1/2)*(D[k-1,k]+D[k,k-1])
    
    # --- Normalize
    #D/=Img.shape[1]

    return D
                          

"""""""""
--- G2 Normalization
"""""""""
def normalize_g2(G2, IF, IP, m, delay, hot_pixel_threshold=1e3):
    """
    Normalize a multi-tau G2 tensor for a single ROI mask and return
    the mean g2 and its standard error across valid pixels.

    Parameters
    ----------
    G2 : ndarray, shape (n_delays, n_pixels)
        Unnormalized G2 tensor from g2calc.
    IF : ndarray, shape (n_delays, n_pixels)
        Future-intensity tensor from g2calc.
    IP : ndarray, shape (n_delays, n_pixels)
        Past-intensity tensor from g2calc.
    m : ndarray, shape (n_pixels,)
        Binary ROI mask (1 = selected pixel).
    delay : ndarray, shape (1, n_delays_expected)
        Delay vector from getdelayinfo.
    hot_pixel_threshold : float, optional
        Pixels whose first-delay G2 value exceeds this are excluded.
        Default is 1e3.

    Returns
    -------
    g2_mean : ndarray, shape (last_index,)
        Mean normalized g2 over valid pixels.
    g2_error : ndarray, shape (last_index,)
        Standard error of normalized g2 over valid pixels.
    last_index : int
        Number of usable delay points.
    """
    G2_masked = G2[:, m == 1]
    hot_pixel_mask = G2_masked[0, :] < hot_pixel_threshold

    if G2_masked.shape[0] == delay.shape[1]:
        last_index = G2_masked.shape[0]
    else:
        last_index = min(G2_masked.shape[0], delay.shape[1])

    part_norm = (np.mean(IF[:, m == 1][:, hot_pixel_mask == 1], axis=1) *
                 np.mean(IP[:, m == 1][:, hot_pixel_mask == 1], axis=1))

    g2dummy = np.divide(
        G2_masked,
        part_norm[:, np.newaxis],
        out=np.full_like(G2_masked, np.nan),
        where=part_norm[:, np.newaxis] != 0
    )

    valid_pixels = hot_pixel_mask == 1
    g2_mean = np.mean(g2dummy[:, valid_pixels][:last_index], axis=1)
    g2_error = (np.std(g2dummy[:, valid_pixels][:last_index], axis=1) /
                np.sqrt(np.sum(valid_pixels)))

    return g2_mean, g2_error, last_index


"""""""""
--- subfunction getdelayinfo
"""""""""
def getdelayinfo(nframes, dpl=4):
    
    # --- get delay and maxlevel
    delay = finddelays(nframes, dpl, 1)
    
    if delay.size <= 2*dpl-1:
        maxLevel = 1
    else:
        maxLevel = int(np.ceil((delay.size - (2*dpl - 1))/dpl) + 1)
        
    return delay, maxLevel


"""""""""
--- subfunction finddelays
"""""""""
def finddelays(nframes, dpl, mindelay=float('nan'), maxdelay=float('nan')):

    # Determine imin index
    if mindelay == 0:
        imin = 1
    else:
        imin = np.fix(np.ceil(indexofdelay(np.array(np.max(mindelay,0))[np.newaxis],dpl)))
    
    # Determine imax index
    imax = np.fix(indexofdelay(np.array(nframes)[np.newaxis],dpl))-1 
    if maxdelay!=float('nan') and type(maxdelay) == int:
        imax = np.min(imax, np.fix(indexofdelay(np.array(maxdelay)[np.newaxis],dpl)))
    
    # Pass indices to delayofindex function
    fDelays = delayofindex(imin,imax,dpl)
    
    #print('fDelays: {}'.format(fDelays))
    return fDelays


"""""""""
--- subfunction indexofdelay
"""""""""
def indexofdelay(delay,dpl):
    
    # Call level of delay
    level = levelofdelay(delay,dpl)
    
    # Initialize lOfDelay
    iOfDelay = np.zeros((1,len(delay)))
    
    iOfDelay[0,delay<dpl] = delay[delay<dpl] + 1
    iOfDelay[0,delay>=dpl] = 1 + dpl*level[0,delay>=dpl]+(delay[delay>=dpl] 
    -dpl*2**(level[0,delay>=dpl]-1))/(2**(level[0,delay>=dpl]-1))
    
    #print('iOfDelay: {}'.format(iOfDelay))
    return iOfDelay


"""""""""
--- subfunction levelofdelay
"""""""""
def levelofdelay(delay,dpl):
    
    lOfDelay = np.zeros((1,len(delay)))
    
    lOfDelay[0,delay<dpl] = 0
    lOfDelay[0,delay>=dpl] = np.ceil(np.log((delay[delay>=dpl]+1)/dpl)/np.log(2))
    
    #print('lOfDelay: {}'.format(lOfDelay))
    return lOfDelay


"""""""""
--- subfunction delayofindex
"""""""""
def delayofindex(imin,imax,dpl):
    # NumPy 2 rejects (1,1) arrays here; these come from indexofdelay
    imin = int(np.asarray(imin).reshape(-1)[0])
    imax = int(np.asarray(imax).reshape(-1)[0])

    # Create Index Vector
    index = np.arange(imin,imax+1,1)

    # Start Index Vector from Zero (not sure with Python)
    index = index - 1
    
    # Initialize Level Vector
    level = np.floor(index/dpl)
    # Initialize dOfIndex Vector
    dOfIndex = np.zeros((1,len(index)))
    #print('index shape {}'.format((index<dpl).shape))
    dOfIndex[0,index<dpl] = index[index<dpl]%dpl
    dOfIndex[0,index>=dpl] = (2**(level[index>=dpl]-1))*(dpl+(index[index>=dpl]%dpl))
    
    #print('dOfIndex: {}'.format(dOfIndex))
    return dOfIndex
    
    