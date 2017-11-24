Search.setIndex({docnames:["eiger_tests","functions","index","instruments","modules","plot","receiver","root_helper","setup","sls_detector_tools","test"],envversion:53,filenames:["eiger_tests.rst","functions.rst","index.rst","instruments.rst","modules.rst","plot.rst","receiver.rst","root_helper.rst","setup.rst","sls_detector_tools.rst","test.rst"],objects:{"":{sls_detector_tools:[9,0,0,"-"]},"sls_detector_tools.AgilentMultiMeter":{measure:[3,2,1,""]},"sls_detector_tools.SourceMeter":{close_port:[3,2,1,""],data_elements:[3,2,1,""],get_digits:[3,2,1,""],off:[3,2,1,""],on:[3,2,1,""],open_port:[3,2,1,""],read:[3,2,1,""],remote:[3,2,1,""],set_digits:[3,2,1,""],set_voltage:[3,2,1,""]},"sls_detector_tools.XrayBox":{HV:[3,2,1,""],get_kV:[3,2,1,""],get_mA:[3,2,1,""],set_kV:[3,2,1,""],set_mA:[3,2,1,""],shutter:[3,2,1,""],target:[3,2,1,""],unlock:[3,2,1,""]},"sls_detector_tools.ZmqReceiver":{get_frame:[6,2,1,""]},"sls_detector_tools.calibration":{do_scurve:[9,3,1,""],do_scurve_fit:[9,3,1,""],do_scurve_fit_scaled:[9,3,1,""],do_trimbit_scan:[9,3,1,""],do_vrf_scan:[9,3,1,""],find_and_write_trimbits:[9,3,1,""],find_and_write_trimbits_scaled:[9,3,1,""],find_initial_parameters:[9,3,1,""],find_mean_and_set_vcmp:[9,3,1,""],find_mean_and_std:[9,3,1,""],generate_calibration_report:[9,3,1,""],generate_mask:[9,3,1,""],get_data_fname:[9,3,1,""],get_fit_fname:[9,3,1,""],get_halfmodule_mask:[9,3,1,""],get_tbdata_fname:[9,3,1,""],get_trimbit_fname:[9,3,1,""],get_vrf_fname:[9,3,1,""],load_trim:[9,3,1,""],per_chip_global_calibration:[9,3,1,""],rewrite_calibration_files:[9,3,1,""],setup_detector:[9,3,1,""],take_global_calibration_data:[9,3,1,""],write_calibration_files:[9,3,1,""]},"sls_detector_tools.config":{Eiger2M:[9,1,1,""],Eiger9M:[9,1,1,""],calibration:[9,1,1,""],path:[9,1,1,""],set_log:[9,3,1,""],tests:[9,1,1,""]},"sls_detector_tools.config.Eiger2M":{beb:[9,4,1,""],hostname:[9,4,1,""]},"sls_detector_tools.config.Eiger9M":{T:[9,4,1,""],beb:[9,4,1,""],hostname:[9,4,1,""]},"sls_detector_tools.config.calibration":{clean_threshold:[9,4,1,""],clkdivider:[9,4,1,""],dynamic_range:[9,4,1,""],energy:[9,4,1,""],exptime:[9,4,1,""],flags:[9,4,1,""],fname:[9,4,1,""],gain:[9,4,1,""],global_targets:[9,4,1,""],nframes:[9,4,1,""],npar:[9,4,1,""],nproc:[9,4,1,""],period:[9,4,1,""],plot:[9,4,1,""],run_id:[9,4,1,""],run_id_trimmed:[9,4,1,""],run_id_untrimmed:[9,4,1,""],std:[9,4,1,""],target:[9,4,1,""],threshold:[9,4,1,""],tp_dynamic_range:[9,4,1,""],tp_exptime:[9,4,1,""],trimval:[9,4,1,""],type:[9,4,1,""],vrf_scan_exptime:[9,4,1,""],vrs:[9,4,1,""],vtr:[9,4,1,""]},"sls_detector_tools.config.path":{base:[9,4,1,""],data:[9,4,1,""],out:[9,4,1,""],test:[9,4,1,""]},"sls_detector_tools.config.tests":{plot:[9,4,1,""],rxb_interval:[9,4,1,""]},"sls_detector_tools.eiger_tests":{analog_pulses:[0,3,1,""],counter:[0,3,1,""],generate_report:[0,3,1,""],io_delay:[0,3,1,""],overflow:[0,3,1,""],plot_lines:[0,3,1,""],rx_bias:[0,3,1,""],setup_test_and_receiver:[0,3,1,""],tp_scurve:[0,3,1,""]},"sls_detector_tools.function":{double_gaus_edge:[1,3,1,""],double_gaus_edge_new:[1,3,1,""],expo:[1,3,1,""],gaus:[1,3,1,""],ideal_dqe:[1,3,1,""],ideal_mtf:[1,3,1,""],paralyzable:[1,3,1,""],pol1:[1,3,1,""],pol2:[1,3,1,""],root:[1,1,1,""],scurve2:[1,3,1,""],scurve4:[1,3,1,""],scurve:[1,3,1,""]},"sls_detector_tools.function.root":{double_gaus_edge:[1,4,1,""],scurve2:[1,4,1,""],scurve4:[1,4,1,""],scurve:[1,4,1,""]},"sls_detector_tools.io":{geant4:[9,1,1,""],load_file:[9,3,1,""],load_frame:[9,3,1,""],load_txt:[9,3,1,""],read_frame:[9,3,1,""],read_frame_header:[9,3,1,""],read_header:[9,3,1,""],read_trimbit_file:[9,3,1,""],save_txt:[9,3,1,""],write_trimbit_file:[9,3,1,""]},"sls_detector_tools.io.geant4":{sparse_dt:[9,4,1,""]},"sls_detector_tools.load_tiff":{load_tiff:[9,3,1,""]},"sls_detector_tools.mask":{eiger2M:[9,1,1,""],eiger500k:[9,1,1,""],eiger9M:[9,1,1,""]},"sls_detector_tools.plot":{add_module_gaps:[5,3,1,""],chip_histograms:[5,3,1,""],draw_module_borders:[5,3,1,""],draw_module_names:[5,3,1,""],fix_large_pixels:[5,3,1,""],global_scurve:[5,3,1,""],imshow:[5,3,1,""],interpolate_pixel:[5,3,1,""],plot_pixel_fit:[5,3,1,""],plot_signals:[5,3,1,""],random_pixels:[5,3,1,""],setup_plot:[5,3,1,""]},"sls_detector_tools.receiver":{ZmqReceiver:[9,1,1,""]},"sls_detector_tools.receiver.ZmqReceiver":{get_frame:[9,2,1,""]},"sls_detector_tools.root_helper":{getHist:[7,3,1,""],hist:[7,3,1,""],plot:[7,3,1,""],style:[7,1,1,""],th2:[7,3,1,""]},"sls_detector_tools.root_helper.style":{hist_color:[7,4,1,""],hist_line_color:[7,4,1,""],hist_line_width:[7,4,1,""],line_color:[7,4,1,""],line_width:[7,4,1,""],marker_color:[7,4,1,""],marker_style:[7,4,1,""]},"sls_detector_tools.utils":{R:[9,3,1,""],generate_scurve:[9,3,1,""],get_dtype:[9,3,1,""],normalize_flatfield:[9,3,1,""],random_pixel:[9,3,1,""],ratecorr:[9,3,1,""],sum_array:[9,3,1,""]},"sls_detector_tools.xray_box":{DummyBox:[9,1,1,""],XrayBox:[9,1,1,""]},"sls_detector_tools.xray_box.DummyBox":{HV:[9,2,1,""],get_kV:[9,2,1,""],get_mA:[9,2,1,""],set_kV:[9,2,1,""],set_mA:[9,2,1,""],shutter:[9,2,1,""],shutter_open:[9,2,1,""],target:[9,2,1,""],unlock:[9,2,1,""]},"sls_detector_tools.xray_box.XrayBox":{HV:[9,2,1,""],get_kV:[9,2,1,""],get_mA:[9,2,1,""],set_kV:[9,2,1,""],set_mA:[9,2,1,""],shutter:[9,2,1,""],target:[9,2,1,""],unlock:[9,2,1,""]},sls_detector_tools:{"function":[1,0,0,"-"],AgilentMultiMeter:[3,1,1,""],SourceMeter:[3,1,1,""],XrayBox:[3,1,1,""],ZmqReceiver:[6,1,1,""],calibration:[9,0,0,"-"],config:[9,0,0,"-"],eiger_tests:[0,0,0,"-"],io:[9,0,0,"-"],load_tiff:[9,0,0,"-"],mask:[9,0,0,"-"],plot:[5,0,0,"-"],receiver:[9,0,0,"-"],root_helper:[7,0,0,"-"],utils:[9,0,0,"-"],xray_box:[9,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"250k":9,"500k":9,"case":[7,9],"class":[1,3,6,7,9],"default":[0,5,7,9],"final":0,"float":9,"function":[2,5,9],"import":[5,9],"int":[3,5,7,9],"new":7,"return":[1,3,5,7,9],"switch":[3,9],"true":[3,5,7,9],"while":[0,9],But:9,The:[0,3,5,7,9],Use:7,Used:9,Uses:9,Using:0,Vrs:9,_calibrate_vrf:9,_fit:9,_rw:9,abov:9,access:[3,9],acq:[6,9],add:[5,7],add_module_gap:5,added:7,afs:[3,9],afs_overwrit:9,agil:[2,4],agilentmultimet:3,aim:7,all:9,allow:9,alp:7,alreadi:9,also:[5,7,9],alwai:9,analog:0,analog_puls:0,anoth:[3,9],anyth:9,appar:7,appli:[5,9],around:[3,9],arrai:[1,5,7,9],array_lik:7,ascii:9,asic:5,asic_color:5,asic_linewidth:5,assum:7,author:9,averag:9,axes:5,axi:[5,7],backend:9,bad:9,base:[0,1,3,6,7,9],baudrat:3,beam:[3,9],beb029:9,beb030:9,beb038:9,beb040:9,beb054:9,beb055:9,beb056:9,beb058:9,beb059:9,beb061:9,beb064:9,beb070:9,beb071:9,beb072:9,beb073:9,beb074:9,beb076:9,beb078:9,beb084:9,beb087:9,beb088:9,beb091:9,beb092:9,beb094:9,beb095:9,beb096:9,beb097:9,beb100:9,beb101:9,beb102:9,beb103:9,beb104:9,beb105:9,beb106:9,beb109:9,beb111:9,beb113:9,beb116:9,beb117:9,beb119:9,beb121:9,beb122:9,beb125:9,beb127:9,beb:9,becaus:[0,9],bee:9,been:9,best:9,better:5,between:5,bia:0,big:[2,4,9],bin:[5,7,9],block:9,blue:0,bname:9,board:9,bool:[3,5,7,9],border:5,both:[0,7,9],bound:9,box:[3,9],build:1,built:9,bytes:3,calibr:[1,4],can:[7,9],cannot:0,canva:7,center:[0,1,7,9],cfg:9,chang:9,check:[0,9],chip:[0,5,9],chip_divid:5,chip_histogram:5,clean_threshold:9,clk:0,clkdivid:9,clock:[0,9],close:[3,9],close_port:3,cmap:5,code:9,col:[5,9],color:[5,7],colorbar:5,colormap:5,colorscal:5,come:9,command:[3,9],commandlin:9,commonli:1,commun:[3,9],complet:9,concern:9,config:[4,5],configur:[7,9],connect:9,consecut:9,consruct:9,contain:9,content:[0,2,4],contin:9,control:[3,9],conveni:[7,9],coolwarm:5,copi:9,corner:5,correct:[3,9],correctli:0,correspond:9,cosmic:0,count:[0,5,9],counter:0,creat:9,csax:9,current:[3,5,6,9],dac:9,dacs8:9,dash:0,data:[0,3,5,6,7,9],data_el:3,data_from_halfmodul:9,data_mask:9,data_typ:9,dead:9,decod:9,defin:9,degre:1,delimit:9,depend:7,design:9,det_id:9,detector:[0,1,5,9],determin:[0,9],dev:3,deviat:9,dict:9,dictionari:9,differ:[7,9],differenti:[5,9],digit:[0,5],direct:[3,9],directori:9,disabl:9,disk:9,displai:5,distribut:9,divid:[0,5,9],do_scurv:9,do_scurve_fit:9,do_scurve_fit_sc:9,do_trimbit_scan:9,do_vrf_scan:9,doe:[0,9],doesn:9,doing:9,don:9,done:0,doubl:[1,5,7,9],double_gaus_edg:1,double_gaus_edge_new:1,dqe:1,draw:[5,7],draw_as:5,draw_module_bord:5,draw_module_nam:5,drawn:[5,7],dtype:9,dummi:9,dummybox:9,dure:9,dynam:9,dynamic_rang:9,each:[0,1,5,9],easi:9,edg:[5,7,9],eiger2m:9,eiger500k:[6,9],eiger9m:[5,9],eiger:[2,9],eiger_test:0,element:[3,9],emul:9,enabl:0,end:[6,9],energi:[1,9],erf:1,error:[3,7,9],error_graph:7,esrf:9,etc:[3,9],evalu:1,even:5,event:9,everyth:9,evri:9,exampl:9,except:7,execut:[3,9],expand:5,expect:[0,5,6,9],expo:1,exponenti:1,exposur:9,exposuretim:9,express:1,expsur:9,exptim:9,extern:7,fals:[0,3,5,7,9],far:0,feel:7,fetch:[5,9],field:[5,9],figsiz:5,figur:5,file:[0,9],fileformat:9,filenam:9,fill:[7,9],fill_styl:7,find:9,find_and_write_trimbit:9,find_and_write_trimbits_sc:9,find_initial_paramet:9,find_mean_and_set_vcmp:9,find_mean_and_std:9,firmwar:9,fit:[5,9],fit_result:[5,9],fix_large_pixel:5,flag:[3,9],flatfield:9,flexibl:9,float64:9,fluoresc:9,fname:9,folder:0,follow:9,font:5,form:9,found:9,fr_fname:9,frame:[5,6,9],frameindex:9,framework:9,from:[1,3,5,6,7,9],fuction:9,full:[0,9],funtion:9,gain:9,gap:5,gau:1,gaussian:1,geant4:9,geant4medipix:9,gener:[0,9],generate_calibration_report:9,generate_mask:9,generate_report:0,generate_scurv:9,geometri:9,geomtetri:9,get:9,get_data_fnam:9,get_digit:3,get_dtyp:9,get_fit_fnam:9,get_fram:[6,9],get_halfmodule_mask:9,get_kv:[3,9],get_ma:[3,9],get_tbdata_fnam:9,get_trimbit_fnam:9,get_vrf_fnam:9,gethist:7,give:[0,9],given:[1,5,9],global:[5,9],global_scurv:5,global_target:9,going:0,graph:7,half:[0,5,9],half_modul:5,halfmodul:9,handi:9,handl:9,hardcod:9,have:[3,7,9],header:[6,9],height:7,helper:2,high:[3,7,9],higher:5,highest:7,highligt:9,hist:7,hist_color:7,hist_line_color:7,hist_line_width:7,histogram:[5,7],hit:7,hold:9,home:9,horizont:9,host:3,hostanm:9,hostnam:9,how:9,ideal:1,ideal_dq:1,ideal_mtf:1,identifi:9,imag:[0,5,6,9],implement:5,imshow:5,includ:9,increment:0,index:[2,5,9],individu:5,inflect:[5,9],inform:9,initi:9,input:[5,7],insert:5,insigt:9,inspect:[3,9],instead:[7,9],instrument:[2,4],int32:9,interfac:[3,9],interpol:5,interpolate_pixel:5,interv:0,io_delai:0,iodelai:0,json:[6,9],just:7,keithlei:[2,4],kept:9,l_frojdh:9,larg:5,larger:[0,9],layout:[5,9],letter:[3,9],level:9,like:[5,7],limit:9,line:[0,5,7,9],line_color:7,line_width:7,linear:1,list:[7,9],load:[0,9],load_fil:9,load_fram:9,load_tiff:4,load_trim:9,load_txt:9,loc:9,lock:3,log:[3,5,9],logaritm:5,logfil:9,logger:[3,9],logic:0,look:[3,9],loos:9,lover:5,low:[7,9],lower:[7,9],lowest:7,main:9,mainli:9,make:[0,3,9],marker_color:7,marker_styl:7,mask:4,master:9,match:3,matplotlib:5,max:5,maximum:9,mean:9,measur:[1,3,9],merg:9,might:9,min:5,miss:9,model:1,modifi:9,modul:[0,2,4,5],module_test:4,mon:9,monochromat:9,more:9,mostli:7,move:9,mpl:5,mppl:5,mtf:1,multi:[5,9],multimet:[2,4],multipag:0,n_photon:9,n_pixel:[5,9],n_puls:0,name:[0,3,5,9],ndac:9,need:[7,9],new_imag:5,nframe:9,nice:5,nois:0,none:[5,7,9],nonparallel:9,normal:[0,7,9],normalize_flatfield:9,note:[0,9],noth:10,notimplementederror:9,nov:9,now:[3,10],np_arrai:9,npar:9,nproc:9,npuls:0,npy:9,npz:9,number:[0,3,5,9],numpi:[5,7,9],numpu_arrai:9,numpy_arrai:[5,9],object:[1,3,5,6,7,9],off:[3,9],offlin:9,old:[0,7,9],older:9,omega:1,one:[5,6,7,9],onli:[0,5,6,9],open:[3,9],open_port:3,oper:3,option:[3,5,7,9],order:1,otherwis:[3,9],out:[0,5,9],outlin:5,output:[0,3,9],over:3,overflow:0,overlai:7,overrid:9,packag:4,page:2,panel:3,par:[5,9],paralyz:1,param:9,paramet:[1,3,5,7,9],pariti:3,part:[0,9],pass:9,patch:0,path:[0,9],pattern:9,pdf:0,per:[0,5,9],per_chip_global_calibr:9,perform:9,performa:9,period:9,physic:3,pixel:[0,1,5,7,9],pixelmask:[5,9],plot:[0,2,7,9],plot_lin:0,plot_pixel_fit:5,plot_sign:5,point:[5,7,9],pol1:1,pol2:1,polynomi:1,port:[3,6,9],possibl:[0,7,9],present:9,print:9,probe:0,problem:0,project:9,provid:[5,9],psi:9,puls:[0,9],pulse_chip:0,put:[0,6,9],pyroot:[2,5],python:[3,7,9],pyton:7,qualiti:9,rai:9,rais:[5,7,9],random:[5,9],random_pixel:[5,9],rang:[0,9],rate:[1,9],ratecorr:9,raw:9,read:[0,3,6,9],read_fram:9,read_frame_head:9,read_head:9,read_trimbit_fil:9,readabl:5,readout:0,realli:9,receiv:[4,6],record:9,red:0,refelct:9,regard:9,reles:9,reli:[5,9],remot:3,remov:9,replac:9,report:[0,9],request:9,requir:[3,9],result:9,rewrit:9,rewrite_calibration_fil:9,right:9,root:[1,5,7],root_help:7,rotat:5,routin:[5,9],row:[5,9],run:[0,9],run_id:9,run_id_trim:9,run_id_untrim:9,rx_bia:0,rxb_interv:9,same:[1,5,7,9],save:9,save_txt:9,scale:[1,5,9],scan:[0,9],script:9,scruve:1,scurv:[1,5,9],scurve2:1,scurve4:1,seaborn:5,search:2,second:1,see:0,select:9,sensor:5,seper:5,serial:3,set:[0,3,5,7,9],set_digit:3,set_kv:[3,9],set_log:9,set_ma:[3,9],set_voltag:3,settingsdir:9,setup:[5,9],setup_detector:9,setup_plot:5,setup_test_and_receiv:0,sever:[5,9],share:9,shift:9,should:[0,5,7,9],show:0,shutter:[3,9],shutter_open:9,side:0,sigl:5,sigma1:1,sigma2:1,sigma:[0,1],signal:5,simpl:[6,9],simul:9,sinc:9,singl:[5,9],size:[5,7,9],skip:[5,9],slice:9,sls_cmodul:9,sls_det_softwar:9,sls_detector:5,sls_detector_put:9,sls_detector_tool:[0,1,3,5,6,7],slsdetector:9,slsdetectorsoftwar:9,slsdetectorspackag:9,sns:5,softwar:[0,9],some:[5,9],someth:7,sourc:[0,1,3,5,6,7,9],sourcemeet:3,sourcemet:[2,4],sparse_dt:9,special:9,specif:9,specifi:[6,7,9],speed:[0,9],spefi:7,spefifi:7,spot:9,sqrt:[1,7],stack:9,standard:[0,9],start:9,state:9,station:0,std:[5,7,9],step:[5,9],stepsiz:9,still:9,stop:9,stopbit:3,store:9,str:[3,5,7,9],stream:[6,9],string:[1,9],structur:9,style:[5,7,9],submodul:4,suffix:9,suit:0,sum:[5,9],sum_arrai:9,sum_siz:9,suppli:9,support:[3,9],sure:[3,9],swap_ax:9,synchrotron:9,system:[5,9],t45:9,t45_vcmp_crxrf_1:9,t_set:[3,9],take:9,take_global_calibration_data:9,taken:[7,9],talk:5,target:[3,9],target_nam:[3,9],tau:[1,9],tb_fname:9,tcanva:7,test:[2,9],textcolor:5,tgraph:7,tgrapherror:7,th1:7,th1d:7,th2:7,th2d:7,thi:[0,5,6,9],thing:7,though:9,thrang:9,threshold:[0,5,9],tiff:9,time:[3,9],timeout:3,titl:7,tmath:1,toa:9,todo:3,togeth:[6,9],toggel:0,toggl:0,tot:9,tp_dynamic_rang:9,tp_exptim:9,tp_scurv:0,tri:9,trim:9,trim_and:9,trimbit:9,trimval:9,trough:9,ttyusb0:3,tube:[3,9],tupl:[5,9],turn:3,two:[3,9],type:[1,3,5,9],typic:9,uint32:9,unfortunatlei:0,unlock:[3,9],updat:9,upper:9,use:[5,9],used:[0,1,5,7,9],useful:9,user:[3,9],uses:[0,3,5,7,9],using:[0,3,5,7,9],usual:5,util:4,uxa:9,v18:9,v19:9,v20:9,v_trim:9,vale:1,valu:[0,1,3,5,7,9],valubl:9,valueerror:[5,7,9],variabl:9,variant:1,variou:9,vcmp:9,vcp:9,verbos:3,verifi:9,version:9,vertic:[5,9],voltag:[3,9],vrf:9,vrf_scan_exptim:9,vrs:9,vtr:9,wai:3,want:9,warn:[5,9],wed:9,well:[5,9],when:[0,9],where:9,which:[7,9],white:5,width:[1,7],wihth:9,window:0,without:[5,9],work:[5,9],wrapper:[3,7,9],write:[5,9],write_calibration_fil:9,write_trimbit_fil:9,wrong:9,xaxi:5,xmax:[5,7],xmin:[5,7],xonxoff:3,xrai:[3,9],xray_box:4,xraybox:[2,4,9],xrayclient64:[3,9],xrf:[3,9],ymax:7,ymin:7,you:[3,9],zmq:[6,9],zmqreceiv:[2,9]},titles:["EIGER Tests","Functions","Welcome to sls_detector_tools\u2019s documentation!","Instruments","sls_detector_tools","Plotting","ZmqReceiver","PyROOT Helper","setup module","sls_detector_tools package","test module"],titleterms:{"function":1,agil:3,big:3,calibr:9,config:9,content:9,document:2,eiger:0,helper:7,indic:2,instrument:3,keithlei:3,load_tiff:9,mask:9,modul:[8,9,10],module_test:9,multimet:3,packag:9,plot:5,pyroot:7,receiv:9,setup:8,sls_detector_tool:[2,4,9],sourcemet:3,submodul:9,tabl:2,test:[0,10],todo:9,util:9,welcom:2,xray_box:9,xraybox:3,zmqreceiv:6}})