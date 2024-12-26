// Load in images from a folder and save Log files as .txt files which 
// show the results of coloc

//clear Log to reset before starting
print("\\Clear");

// choose a file from a directory to open in ImageJ
path = File.openDialog("Select a File");
//open(path); // open the file
dir = File.getParent(path);
name = File.getName(path);
print("Path:", path);
print("Directory:", dir);
print("Name:", name);
open(path); //file will open in ImageJ - here manually select Hyperstack and select all and OK

imgs = getList("image.titles");

for (i = 0; i < imgs.length; i++) {
   print("Processing image: "+imgs[i]);
   selectImage(imgs[i]);
   imageName = imgs[i];
   // split channels into different fluorescence channels
   run("Split Channels");
   title = getTitle();
   print("Title:", title);
   ch1 = "C1-"+imageName;
   print(ch1);
   ch2 = "C2-"+imageName;
   print(ch2); 
   run("Coloc 2", "channel_1=[&ch1] channel_2=[&ch2] roi_or_mask=<None> threshold_regression=Costes li_histogram_channel_1 li_histogram_channel_2 li_icq spearman's_rank_correlation manders'_correlation kendall's_tau_rank_correlation 2d_intensity_histogram costes'_significance_test psf=3 costes_randomisations=10");
   
}

savePath = path+"_Log.txt";
print(savePath)
selectWindow("Log");
saveAs("Text", "&savePath"); //saving doesn't seem to work??

close("*")
// would be nice to also close coloc channels although haven't worked out how to do this yet
