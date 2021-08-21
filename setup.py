from setuptools import setup, find_namespace_packages

setup(name='nnformer',
      packages=find_namespace_packages(include=["nnformer", "nnformer.*"]),
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'nnFormer_convert_decathlon_task = nnformer.experiment_planning.nnFormer_convert_decathlon_task:main',
              'nnFormer_plan_and_preprocess = nnformer.experiment_planning.nnFormer_plan_and_preprocess:main',
              'nnFormer_train = nnformer.run.run_training:main',
              'nnFormer_train_DP = nnformer.run.run_training_DP:main',
              'nnFormer_train_DDP = nnformer.run.run_training_DDP:main',
              'nnFormer_predict = nnformer.inference.predict_simple:main',
              'nnFormer_ensemble = nnformer.inference.ensemble_predictions:main',
              'nnFormer_find_best_configuration = nnformer.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnFormer_print_available_pretrained_models = nnformer.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnFormer_print_pretrained_model_info = nnformer.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnFormer_download_pretrained_model = nnformer.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnFormer_download_pretrained_model_by_url = nnformer.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnFormer_determine_postprocessing = nnformer.postprocessing.consolidate_postprocessing_simple:main',
              'nnFormer_export_model_to_zip = nnformer.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnFormer_install_pretrained_model_from_zip = nnformer.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnFormer_change_trainer_class = nnformer.inference.change_trainer:main',
              'nnFormer_evaluate_folder = nnformer.evaluation.evaluator:nnFormer_evaluate_folder',
              'nnFormer_plot_task_pngs = nnformer.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      
      )
