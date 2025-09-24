# Estimating of Gating Latencies with an Elekta Linac and ExacTrac Dynamic

## Project status
Project is in the very early stage. Use with caution and interpret your results with care.
The code, as presented here, contains the analysis of end-to-end (E2E) latencies. Soon, the code to analyze the gating system latencies, will also be published. Though, the corresponding manuscript is in development and to be expected for 2026. Then, we will upload the full code base to analyze the gating system's latencies, too.
If you want to analyze the gating system latencies of your treatment unit now, please contact Catrin for further support.

## Description
This project uses videos and photographs to estimate the gating latency of SGRT-gating systems.
![Workflow](screenshots_readme/workflow_whole-loop_latencies_paper1.png)
![DIBH cycles](screenshots_readme/graphic_DIBH_zyklen.png)

## Installation & Testing
1. Download or clone the code from this repository
2. Download the [test data](https://drive.google.com/drive/folders/1FIbJGtuY2rg9T6_IJBwp3XNRX0UpYVXH?usp=sharing) and move the folder "data_test_DIBH_vid" to "testcases > test_DIBH_vid"
3. Install Python on your device. This code has been created and tested using Python 3.11.2. You can probably run it with older Python versions, too.
4. Create a working environment (see requirements). At best, create a virtual environment first. For example, creating a virtual environment called "gatlat" with venv:
```shell
$ <your-python-command> -m venv path\to\venvs\gatlat
```
Secondly, install the requirements:
```shell
$ pip install -r path/to/this/repository/requirements.txt
```
(Testing your installation is not needed if you only want to reproduce the results of our publication. However, if you encounter any inconsistencies with the results of our publications, make sure to run the tests first. In general, I strongly recommend test any installation before running any analyses.)

5. Test your installation by first, running the in-silico test case on non-video data:
```shell
$ <your-python-command> analyze_pixel_in_ROI_static.py config_test_DIBH_noVid.json
```
The resulting beam-on latencies should all evolve around 2000 ms \pm 8.3 ms (1 frame). The resulting beam-off latencies should all evolve around 200 ms \pm 8.3 ms (1 frame).
- [ ] insert screenshot

6. Run an analysis of the test video (1 reference step pattern, 3 DIBH phases). This takes time, be patient. When the code is looping over each single frame of the video, a progress bar indicates your waiting time.
```shell
$ <your-python-command> analyze_pixel_in_ROI_static.py config_test_DIBH_vid.json
```
The results should read as:
![Screenshot of results for test video.](screenshots_readme/Screenshot_testCase_vid.png)
(A proper testing suite is under development.)

## Reproduce the results of our Poster at [ECMP 2024](https://ecmp2024.org/)
1. Run the installation and testing process as above.
2. Make sure to access the commit with the right tag "ECMP2024"
3. Download [data_ECMP2024](https://drive.google.com/drive/folders/1FIbJGtuY2rg9T6_IJBwp3XNRX0UpYVXH?usp=sharing) and move it to "data_pub".
4. Make sure data_pub > data_ECMP2024 is part of your repository
5. Adjust the absolute paths in the respective config files "config_video_recXXX_ECMP_results.json"(3 paths per file) before execution
6. run 
```shell
$ <your-python-command> analyze_pixel_in_ROI_static.py config_video_rec1_ECMP_results.json
$ <your-python-command> analyze_pixel_in_ROI_static.py config_video_rec2_ECMP_results.json
$ <your-python-command> analyze_pixel_in_ROI_static.py config_video_rec3_ECMP_results.json
```
If you are interested in the details of the analysis, enable additional plotting feautures in the section "developerSettings" of the config.json file, for example set "intermediate_plotting_of_reference" to "true" to plot the determination of the lower gating level.

## Usage for quality assurance (QA)
### Clinical Remark
When performing quality assurance of medical products and for medical use, always scrutinize the results and the methods producing them. Never apply a method blindly, especially when it is fresh on the market. Use it with caution!

### Procedure
So far, we have only tested the method at Elekta Versa HD linac, equipped with ExacTrac Dynamic for gating. In general, we expect to irritation when applying the method at linacs of other vendors or with other SGRT-systems. But this has not been tested, yet.
Required materials:
* Plastic or any other scintillator irradiating on the blue spectrum (If other visible color, adapt the code in the respective lines.)
* Action camera with high frame rate and sufficient SD card (adapted from [1]). We use a camera with 120 FPS. Frame rate can be set in the config-file. Smaller frame rate increases the uncertainties of the measurements.
* Motion phantom with surface. We use the CIRS Dynamic Motion Phantom (Sun Nuclear) with a thermosplastic mask mimicking a thorax surface. The phantom of your choice, does not necessarily cover a surface as large as a human thorax. The surface must be large enough to work well with the SGRT/linac setup at hand.
* A motion pattern featuring a step function to detect the lower gating level and a DIBH pattern. Must be provided in the data format suitable for the phantom of choice. In DATA you can find example files (.txt and .png). The reference pattern at the beginning is crucial to detect the lower gating level. Otherwise, no analysis possible with this method.
* Background screen to create a smooth background.
  
![Experimental_Setup_in_treatment_room.](screenshots_readme/experimental_setup_treatment_room.jpg)

Data Acquisition:
1. In a standard bright room ambience, setup the screen in the background, setup the phantom. For ETD we do this in free breathing.
2. Position the phantom with your SGRT system. For ETD, we do this in DIBH.
3. Start the phantom motion, start the camera recording (at best, use remote steering of the camera).
4. From the control room, start the irradiation procedure. Beam must be gated by the breathing of the phantom.
5. After irradiation, stop the camera. 
Video Analysis:
6. At the computer, install the code and environment as described above.
7. Move the data to a computer, adapt the config-file (date, any notes, provide the file location and any other information). "region" and "exclude" can be left empty for now and set during analysis (the software will guide you what to insert when):
Example for setting regions during analysis:
```json
"videos" : { ...
            "video1": {
                ...
                "region" : [[]], # "[[]]" indicates that you are going to set the region of interest during the analysis (in terminal)
                "exclude": [[]],
                ...},
            ...}
```

Example nothing to exclude / no irregularities during data acquisition:
```json
"videos" : { ...
            "video1": {
                ...
                "region" : [29001,62139], # [first frame, last frame]
                "exclude": [0], #"[]" indicates that you do not wish to exclude any regions from the analysis
                ...},
            ...}
```

8. Start the analysis with the following command
```shell
$ <your-python-command> analyze_pixel_in_ROI_static.py <your-config-file-name>.json
```

## Roadmap
Future work will incorporate the assessment of separate latencies of the gating system and the linac. This work has been share at ESTRO2025 and DGMP2025. Code will follow after publication of manuscript in 2026.

## Contributing
Are you interested in working on gating latencies of SGRT-systems or want to collaborate? Contact Catrin for details: catrin.rodenberg@med.uni-muenchen.de

## References
[1] Worm, E.S. et al. (2023). Medical Physics, 50(6), 3289-3298; 
[2] Rodenberg, C. et. a. (2025) Radiotherapy and Oncology, 206:S3685-S3686 (abstract ESTRO poster), DOI:10.1016/S0167-8140(25)02008-0

## License
This code employs software that is licensed with Apache License 2.0 (openCV), BSD 3-clause "new" or "revised" license (scipy), and MIT. Your are allowed to use this code for your private or research activities. You must not sell parts of this code. Please get in contact with us, once using it, so we can keep track of it.

