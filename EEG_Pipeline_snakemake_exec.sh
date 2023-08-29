#!/bin/bash
#Cathrine Petersen, Syon Mansur 07/18/2023
#EEG_Pipeline_snakemake_exec.sh
#
# See README.md for information on how to run this pipeline.
#
##############################################################################################

#GLOBAL VARIABLES:

#submission strings for snakemake --cluster parameter
QSUB="--jobs 64 --cluster "\"'qsub -cwd -pe smp {threads} -l mem_free={resources.mem_qsub} -l h_rt={resources.job_time} -m bea -M cathrine.petersen@gladstone.ucsf.edu -j yes -V -o ${LOG_DIR} -e ${LOG_DIR}'\"
SLURM="--jobs 64 --cluster "\"'sbatch -p sfgf,normal --cpus-per-task={threads} --mem={resources.mem_slurm} --time={resources.job_time} --output={params.log_file}'\"
LOCAL="--cores 16"
##############################################################################################

#PIPELINE VARIABLES:

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )" #retrieve the source directory of this pipeline script
CLUSTER_SUB=${QSUB}
IN_FLAG=0 #flag to track if input directory was provided. if not, pipeline will fail because this is a required argument
OUT_FLAG=0 #flag to track if an output directory was provided. if not, pipeline will fail because this is a required argument
BATCH_FLAG=0 #flag to track if a batch id was provided. if not, pipeline will fail because this is a required argument
TEMP_DIR=~/temp/EEG_processing
FORCE=1 #flag to track if temporary directory checks can be ignored (risks overwriting)
REMOVE=0 #flag to track whether the temporary directory should be removed at the end of the pipeline run

##############################################################################################

#COMMAND LINE OPTION INTERPRETATION:
#Handle command line inputs using getopts
while getopts ":hFNRi:s:x:o:d:b:" opt; do
	case $opt in
	h)
		echo -e "\nUsage: bash /path/to/EEG_Pipeline_snakemake_exec.sh -m <Input_Manifest> -o <Output_Directory> ... <other options>\n"
		echo -e "Check the README.md file for detailed usage instructions!"
		exit 0
		;;
	i)
		if [[ -d ${OPTARG} ]]; then
			INPUT_DIR=${OPTARG}
			IN_FLAG=1
			echo "-i flag observed. INPUT_DIR set to ${OPTARG}." >&2
		else
			echo "ERROR --- -i flag observed but suggested INPUT_DIR does not exist: ${OPTARG}" >&2
			exit 1
		fi
		;;
	s)
		if [[ -d ${OPTARG} ]]; then
			SCRIPT_DIR=${OPTARG}
			echo "-s flag observed. Replacing default SCRIPT_DIR with ${OPTARG}." >&2
		else
			echo "ERROR --- -s flag observed but suggested SCRIPT_DIR does not exist: ${OPTARG}" >&2
			exit 1
		fi
		;;
	x)
		echo "-x flag observed with ${OPTARG} option." >&2
		if [ ${OPTARG} = "slurm" ]; then
			CLUSTER_SUB=${SLURM}
		elif [ ${OPTARG} = "qsub" ]; then
			CLUSTER_SUB=${QSUB}
		elif [ ${OPTARG} = "none" ]; then
			CLUSTER_SUB=${LOCAL}
		else
			echo "ERROR --- -x flag observed but specified cluster submission does not match either qsub, slurm, or none" >&2
			exit 1
		fi
		;;
	o)
		OUTPUT_DIR=${OPTARG}
		mkdir -p ${OUTPUT_DIR}
		OUT_FLAG=1
		echo "-o flag observed. OUTPUT_DIR set to ${OPTARG}." >&2
		;;
	d)
		TEMP_DIR=${OPTARG}
		echo "-d flag observed. Replacing default TEMP_DIR with ${OPTARG}." >&2
		;;
	b)
		BATCH_ID=${OPTARG}
		BATCH_FLAG=1
		echo "-b flag observed. BATCH_ID set to ${OPTARG}." >&2
		;;
	F)
		echo "-F flag observed. Temporary alignment directory may be overwritten!" >&2
		FORCE=0
		;;
	N)
		echo "-N flag observed. Temporary alignment directory will not be deleted at the end of the run!" >&2
		RSYNC=1
		;;
	\?)
		echo "Invalid option: -$OPTARG. Check the README.md file for detailed usage instructions!" >&2
		exit 1
		;;
	:)
		echo "Option -$OPTARG requires an argument. Check the README.md file for detailed usage instructions!" >&2
		exit 1
		;;
	*)
		echo "Unimplemented option: -$OPTARG. Check the README.md file for detailed usage instructions!" >&2
		exit 1
		;;
	esac
done

#Check for presence of input and output directory from command line input
if [ "$IN_FLAG" != 1 ]; then
	echo "No valid input directory supplied. Check -i argument!"
	exit 1
fi
if [ "$OUT_FLAG" != 1 ]; then
	echo "No output directory provided. -o argument is required!"
	exit 1
fi
if [ "$BATCH_FLAG" != 1 ]; then
	echo "No Batch ID provided. -b argument is required!"
	exit 1
fi


#check for validity of TEMP_DIR
#if FORCE is FALSE, check if directory exists. if so quit.
if [[ "$FORCE" -eq 1 ]]; then
	if [[ -d ${TEMP_DIR} ]]; then
		echo "ERROR --- TEMP_DIR already exist: ${TEMP_DIR}" >&2
		echo "Pipeline expects that the temporary directory does not exist to prevent overwriting of data or collision of multiple pipeline runs" >&2
		echo "Either (i) use the -F flag to override this, (ii) delete the suggested directory, or (iii) provide a path to a directory that does not already exist using the -d argument." >&2
		exit 1
	else
		mkdir -p ${TEMP_DIR}
	fi
else
	mkdir -p ${TEMP_DIR}
fi

#Fix all provided directory paths to remove trailing forward slashes ("/"). This prevents snakemake from complaining about double slashes ("//") in paths.
INPUT_DIR=$(echo "$INPUT_DIR" | sed 's:/*$::')
TEMP_DIR=$(echo "$TEMP_DIR" | sed 's:/*$::')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | sed 's:/*$::')

echo -e "\n\n"
echo "Input Directory: ${INPUT_DIR}"
echo "Temporary Directory: ${TEMP_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"

#pause for user to see pipeline messages
sleep 2

##############################################################################################

#Check input directory -- should contain a metadata file and video file named according to designated video name 

#new
VIDEO_NAME=$(echo ${BATCH_ID//.*/})
VIDEO_FILE_PATH=${INPUT_DIR}/Raw_EEG_Videos/${BATCH_ID}.wmv
METADATA_FILE=${INPUT_DIR}/${BATCH_ID}_metadata.csv

echo -e "\n\n"
echo "Video File: ${VIDEO_FILE_PATH}"
echo "Video Name: ${VIDEO_NAME}"
echo "Metadata File: ${METADATA_FILE}"
echo -e "\n\n"

#pause for user to see pipeline messages
sleep 2

# Check if video file exists
# if [ -f "$VIDEO_FILE_PATH" ]; then
# 	echo -e "Video file exists. Proceeding...\n"
#else
#	echo "ERROR --- Video file not found. Exiting." >&2
#	exit 1
#fi

# Check if metadata file exists
if [ -f "$METADATA_FILE" ]; then
	echo -e "Metadata file exists. Proceeding...\n"
else
	echo "ERROR --- Metadata file not found. Exiting." >&2
	exit 1
fi

#Designate a directory within TEMP_DIR to store log files
LOG_DIR=${TEMP_DIR}/logs
mkdir -p ${LOG_DIR}

#change directory to the temporary directory
cd ${TEMP_DIR}

#Run snakemake command
COMMAND="snakemake --rerun-incomplete --snakefile ${SCRIPT_DIR}/EEG_Pipeline_snakemake.py --config in_dir=${INPUT_DIR} out_dir=${OUTPUT_DIR} \
temp_dir=${TEMP_DIR} batch_id=${BATCH_ID} scripts=${SCRIPT_DIR} --keep-going ${CLUSTER_SUB}"

echo $COMMAND

eval ${COMMAND}

if [ $? -eq 0 ]; then
	if [[ "$REMOVE" -eq 0 ]]; then
		echo -e "\n\n##########################################################################################################"
		echo Snakemake completed successfully! Removing temporary directory.
		echo -e "##########################################################################################################\n\n"
		rm -r ${TEMP_DIR}
	else
		echo -e "\n\n##########################################################################################################"
		echo Snakemake completed successfully! -N flag observed so temporary directory will not be removed.
		echo -e "##########################################################################################################\n\n"
	fi
else
	echo -e "\n\n##########################################################################################################"
	echo Snakemake failed. Temporary directory is left in place.
	echo -e "##########################################################################################################\n\n"
fi









