C***********************************************************************
C  Propagacion de paquetes de onda en 1 dimension.
C  -----------------------------------------------
C  El programa esta basado en el programa de S. Gray que me paso Ned.
C  Este a su vez esta basado en el 'Kosloff-Kosloff (exact) propagation
C     scheme'.
C  Ver J. Chem. Phys. 92, 2087 (1988).
C
C  Tino, Abril 1994
C  Se perdio y lo volvi a hacer en Sept. 2001
C***********************************************************************
       implicit real*8 (a-h,o-z)
       complex*16 psi(256),psil(256),del(256),dt2
       complex*16 a,hpsi,psic,w1(256),w0(256)
       dimension akx2(256),v(256),psi2(256)
       common/parm/rmu,om,x0
       common/sur/rmax
C
       open(10,file='wavef.dat')
C-----------------------------------------------------------------------
C  Parametros para un oscilador armonico
C-----------------------------------------------------------------------
       rmu=1.d0
       rk=1.d0
       w=rmu
       om=dsqrt(rk/rmu)
C-----------------------------------------------------------------------
C  The grid is from X0 to X0+(NPOINX-1)*DX
C  ITIM = fraction of vibrational period for TIME STEP
C  NTAU = no. time steps of TAU
C   Every NLOOK of the small TIME STEPS is output
C-----------------------------------------------------------------------
       read(5,*) x0,dx
       read(5,*) itim,ntau,npoinx,nlook
       read(5,*) rmax
C
       write(6,*) ' X0, DX',x0,dx
       xfin=x0+(npoinx-1)*dx
       write(6,*) ' XFIN',xfin
       write(6,*) ' Cut-off for PSUR integral, BIGR = ',rmax
       write(6,*) ' ITIM, NTAU, NPOINX ',itim,ntau,npoinx
       write(6,*) ' NLOOK',nlook
C
       tau=2.d0*3.14159265d0/om
       write(6,*) ' TAU ',tau
       dt=tau/dfloat(itim)
       write(6,*) ' DT ',dt
C
       time=0.d0
       call pot(time,v,dx,npoinx)
       call psinit(psil,dx,npoinx)
       write(10,*) time
       x=x0-dx
       do 10 i=1,npoinx
        x=x+dx
        write(10,*) x,cdabs(psil(i))**2
10     continue
C
C  Evaluate t=0 properties
C
       cw=1.d0/2.d0/rmu
       call initak(akx2,npoinx,dx)
C
       call prefft(npoinx,1,nexp,w1)
       call prefft(npoinx,0,nexp,w0)
       call diff(nexp,w1,w0,psil,del,akx2,npoinx)
       energ=0.d0
       hnorm=0.d0
       do 100 k=1,npoinx
          a=psil(k)
          psic=dconjg(psil(k))
          hpsi=(del(k)*cw-v(k)*psil(k))
          energ=energ-psic*hpsi*dx
          psi2(k)=psic*a
          hnorm=hnorm+psi2(k)*dx
100    continue
       write(6,*)
       write(6,*) ' TIME ',time,'; HNORM ',hnorm
       write(6,*) ' ENERGY ',energ
C
C  First integration step
C
       call rung(nexp,w1,w0,psi,psil,del,akx2,v,w,dt,dx,npoinx)
       dt2=(0.d0,1.d0)*2.d0*dt
       time=dt
       call surfin(psi,probt,dx,npoinx)
       write(6,*) ' TIME',time,'; PROBT',probt
       write(6,*)
1000   format(1X,6d14.6)
       write(6,1001)
1001   format(8X,'TIME',9X,'HNORM',9X,'ENERGY',8X,'PROBT',
     *        8X,'LOG(PROB)'/3X,68('-'))
       ll=1
       nstep=itim*ntau
C  Remaining time steps
       do 200 istep=1,nstep
         call diff(nexp,w1,w0,psi,del,akx2,npoinx)
         call pot(time,v,dx,npoinx)
         energ=0.d0
         hnorm=0.d0
         do 210 n=1,npoinx
          a=psi(n)
          hpsi=(del(n)*cw-v(n)*psi(n))
          if(ll.ne.nlook) go to 220
          psic=dconjg(a)
          psi2(n)=psic*a
          hnorm=hnorm+psi2(n)*dx
          energ=energ-psic*hpsi*dx
220       continue
C  Main propagation step:
          psi(n)=psil(n)+dt2*hpsi
          psil(n)=a
210      continue
         time=time+dt
         if(ll.ne.nlook) go to 230
           ll=0
           call surfin(psi,probt,dx,npoinx)
           rlog=dlog(probt)
           write(6,1000) time,hnorm,energ,probt,rlog
           write(10,*)time
           x=x0-dx
           do 235 i=1,npoinx
           x=x+dx
235        write(10,*) x,cdabs(psi(i))**2
230      continue
         ll=ll+1
200    continue
       stop
       end
C-----------------------------------------------------------------------
       subroutine psinit(psi0,dx,n)
C
C  Calculates initial wavefunction
C
       implicit real*8(a-h,o-z)
       common/parm/rmu,om,x0
       complex*16 ri
       complex*16 psi0(n)
C
       ri=(0.d0,1.d0)
       a=rmu*om
       cte=dsqrt(dsqrt(a/3.14159265d0))
       x=x0-dx
       do 100 i=1,n
         x=x+dx
         psi0(i)=cte*dexp(-a*x*x/2.d0)*cdexp(ri*2.d0*x)
100    continue
       return
       end
C-----------------------------------------------------------------------
       subroutine surfin(psi,prob,dx,npoinx)
C
C  Calculates: int (Psi^* Psi dx)
C
       implicit real*8(a-h,o-z)
       complex*16 psi(npoinx)
       complex*16 ampltd,fa,fb
       common/parm/rmu,om,x0
       common/sur/rmax
       ampltd=(0.d0,0.d0)
       fa=dconjg(psi(1))*psi(1)
       bigr=x0
       do 100 i=2,npoinx
         bigr=bigr+dx
         if(dabs(bigr).gt.dabs(rmax)) go to 999
         fb=dconjg(psi(i))*psi(i)
         ampltd=ampltd+(fa+fb)*dx/2.d0
         fa=fb
100    continue
999    continue
       prob=dconjg(ampltd)*ampltd
       return
       end
C-----------------------------------------------------------------------
       subroutine initak(ak,n,delta)
       implicit real*8(a-h,o-z)
       dimension ak(0:n-1)
C
       pi=4.d0*datan(1.d0)
       anorm=2.d0*pi/(dfloat(n)*delta)
C
C  Initialize array
C
       ak(0)=0.d0
       do 100 j=1,n/2
         temp=(dfloat(j)*anorm)**2
         ak(n-j)=-temp
         ak(j)=ak(n-j)
100    continue
       return
       end
C-----------------------------------------------------------------------
       subroutine rung(nexp,w1,w0,psi,psil,del,akx2,v,w,dt,dx,npoinx)
       implicit real*8(a-h,o-z)
       complex*16 psi(npoinx),psil(npoinx),del(npoinx),en
       complex*16 w1(1),w0(1)
       dimension akx2(npoinx),v(npoinx)
       call diff(nexp,w1,w0,psil,del,akx2,npoinx)
       do 2 n=1,npoinx
       en=-(del(n)/(2.d0*w)-v(n)*psil(n))
       psi(n)=psil(n)-(0.d0,1.d0)*0.5d0*dt*en
       del(n)=psil(n)+
     *        (0.d0,1.d0)*dt*(del(n)/(2.d0*w)-v(n)*psi(n))
2      continue
       call diff(nexp,w1,w0,del,del,akx2,npoinx)
       do 3 n=1,npoinx
       psi(n)=psi(n)+(0.d0,1.d0)*0.5d0*dt*(del(n)/(2.d0*w)
     *        -v(n)*psil(n))
3      continue
       return
       end
C-----------------------------------------------------------------------
       subroutine diff(nexp,w1,w0,psi,dpsi,akx2,nx)
C
C  Determines NABLA^2 using FFT method
C
       implicit real*8(a-h,o-z)
       complex*16 w1(1),w0(1)
       complex*16 psi(nx),dpsi(nx)
       dimension akx2(nx)
C
       do 100 j=1,nx
100    dpsi(j)=psi(j)
       call fft(nx,1,nexp,w1,dpsi)
       do 200 ix=1,nx
200    dpsi(ix)=dpsi(ix)*akx2(ix)/dfloat(nx)
       call fft(nx,0,nexp,w0,dpsi)
C
C
       return
       end
C-----------------------------------------------------------------------
       subroutine pot(time,v,dx,npoinx)
       implicit real*8(a-h,o-z)
       common/parm/rmu,om,x0
       dimension v(npoinx)
C
C Harmonic potential
C
      x=x0-dx
      do 100 i=1,npoinx
        x=x+dx
        v(i)=x*x/2.d0
100    continue
       return
       end
C-----------------------------------------------------------------------
       subroutine prefft(n,mode,nexp,w)
       implicit real*8(a-h,o-z)

C
C  Input parameters:
C
C    N     - Number of data samples to be processed (Integer; must be
C            a power of 2)
C    MODE  - Set to 0 for discrete-time Fourier series (eq. 2.c.1)
C            or 1 for inverse (eq. 2.c.2)
C    NEXP  - Indicates power-of-2-exponents such that N=2**NEXP.
C            Will be set to -1 to indicate error condition if N is
C            not a power of 2 (this integer used by subroutine FFT)
C    W     - COMPLEX*16 exponential array
C
       complex*16 w(1),c1,c2
       nexp=1
50     nt=2**nexp
       if(nt.ge.n) go to 100
        nexp=nexp+1
        go to 50
100    if(nt.eq.n) go to 150
        nexp=-1
        return
150    s=8.d0*datan(1.d0)/dfloat(nt)
       c1=dcmplx(dcos(s),-dsin(s))
       if(mode.ne.0) c1=dconjg(c1)
       c2=(1.d0,0.d0)
       do 200 k=1,nt
        w(k)=c2
200    c2=c2*c1
       return
       end
C-----------------------------------------------------------------------
       subroutine fft(n,mode,nexp,w,x)
       implicit real*8(a-h,o-z)
C
C  Input parameters:
C
C    N,MODE,NEXP,W - See parameter list in SUBROUTINE FFT.
C    T             - Sample interval in seconds
C    X             - Arraay of N COMPLEX*16 data samples, X(1) to X(N)
C
C  Output parameters:
C
C    X - N COMPLEX*16 transform values replace original data samples
C        indexed fron K=1 to K=N, representing the frequencies
C        (K-1)/NT hertz
C
C  External array X must be dimensiones .GE. N by calling program
C
       complex*16 x(1),w(1),c1,c2
       mm=1
       ll=n
       do 70 k=1,nexp
        nn=ll/2
        jj=mm+1
        do 40 i=1,n,ll
         kk=i+nn
         c1=x(i)+x(kk)
         x(kk)=x(i)-x(kk)
40      x(i)=c1
        if(nn.eq.1) go to 70
        do 60 j=2,nn
         c2=w(jj)
         do 50 i=j,n,ll
          kk=i+nn
          c1=x(i)+x(kk)
          x(kk)=(x(i)-x(kk))*c2
50       x(i)=c1
60      jj=jj+mm
       ll=nn
       mm=mm*2
70     continue
       nv2=n/2
       nm1=n-1
       j=1
       do 90 i=1,nm1
        if(i.ge.j) go to 80
        c1=x(j)
        x(j)=x(i)
        x(i)=c1
80      k=nv2
85      if(k.ge.j) go to 90
        j=j-k
        k=k/2
        go to 85
90     j=j+k
C
C      if mode.eq.0 s=t
C      if mode.ne.0 s=1.d0/(t*dfloat(n))
C      do 100 i=1,n
C 100  x(i)=x(i)*s
C
       return
       end
