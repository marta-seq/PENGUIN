import os
import numpy as np
from glob import glob

from matplotlib import pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import interact, widgets

import src.ImagePreprocessFilters as IPrep
import src.ImageParser as IP
import src.jupyter_functions as JN


class ImageProcessingWidget:
    def __init__(self):
        self.output_widget = widgets.Output()
        self.plot_output_histogram = widgets.Output()
        self.plot_output_images = widgets.Output()
        self.plot_output_comparison = widgets.Output()
        self.plot_output_ZOOM = widgets.Output()
        self.plot_output_psnr = widgets.Output()

        self.create_widgets()

    def create_widgets(self):
        self.PATH = widgets.Text(value='data/IMC_ESD/raw', description='Path:', disabled=False)
        self.button = widgets.Button(description="Change Path")
        self.button.on_click(self.update_output)

        self.txtbox = widgets.Text(value='ResultsPercentile/', placeholder='Type something',
                                   description='PathSave:', disabled=False)
        self.btn = widgets.Button(description='Save')
        self.btn_bin = widgets.Button(description='Save Binary')
        self.btn.on_click(self.btn_eventhandler)
        self.btn_bin.on_click(self.btn_bin_eventhandler)

        self.save_btn = widgets.HBox([self.btn, self.btn_bin])

        self.result = []
        self.channel_names = []
        self.files_channel = []
        self.files_by_sample = {}

        display(self.PATH)
        display(self.button)
        display(self.output_widget)

    def set_params(self, res=300):
        plt.rcParams['figure.dpi'] = res
        plt.rcParams['savefig.dpi'] = res

    def update_output(self, button):
        with self.output_widget:
            clear_output(wait=True)
            path = self.PATH.value
            if path:
                self.load_files_from_folder(path)
                if self.result:
                    print(f"Number of images identified: {len(self.result)}")
                    self.create_image_widgets()

    def load_files_from_folder(self, folder_path):
        try:
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"{folder_path} is not a valid directory.")

            self.result = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
            num_images = len(self.result)
            print(f"Number of images identified: {num_images}")

            self.channel_names = set([name.split("_")[-1].split(".ome.tiff")[0] for name in self.result])

            if self.channel_names:
                if isinstance(self.channel_names, list) and all(isinstance(name, str) for name in self.channel_names):
                    print("Channel names provided.")
                else:
                    print("Using index-based channel selection.")
            else:
                print("No channel names provided. Using index-based channel selection.")

        except Exception as e:
            print(f"Error: {e}")
            self.result = []

    def create_image_widgets(self):
        self.dropdown_ch_name = widgets.Dropdown(options=self.channel_names, description='Channel:')

        self.dropdown_th = widgets.Dropdown(options=[None, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
                                                     'otsu', 'isodata', 'Li', 'Yen', 'triangle', 'mean', 'local'],
                                            description='Threshold:')
        self.dropdown_percentile = widgets.Dropdown(
            options=[None, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 5, 95, 'p50consecutive'],
            description='Percentile:')
        self.dropdown_sample_images_number = widgets.Dropdown(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                              description='N imgs:')
        self.dropdown_sample_images = widgets.Dropdown(options=['top', 'bottom', 'random', 'all'],
                                                       description='Subset:')
        self.dropdown_high_constrast_compare = widgets.Dropdown(options=['False', 'True'],
                                                                description='HighContrast:')
        self.dropdown_res = widgets.Dropdown(options=[50, 100, 200, 300, 400, 500], description='Resolution:')

        self.dropdown_image_names = widgets.Dropdown(options=['None'],
                                                     description='Image:')

        self.dropdown_ch_name.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_th.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_percentile.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_sample_images_number.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_sample_images.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_high_constrast_compare.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_res.observe(self.dropdown_eventhandler, names='value')
        self.dropdown_image_names.observe(self.dropdown_eventhandler, names='value')


        input_widgets = widgets.HBox([self.dropdown_ch_name, self.dropdown_percentile, self.dropdown_th])

        display_statement = widgets.Label(value='Display:')
        display_statement.layout.margin = '0px 0px 5px 0px'

        display_dropdowns = widgets.HBox([self.dropdown_sample_images_number, self.dropdown_sample_images,
                                          self.dropdown_high_constrast_compare, self.dropdown_res,
                                          self.dropdown_image_names])

        display_layout = widgets.VBox([display_statement, display_dropdowns])

        tab = widgets.Tab([self.plot_output_images, self.plot_output_histogram, self.plot_output_comparison,
                           self.plot_output_ZOOM, self.plot_output_psnr])

        tab.set_title(0, 'Images')
        tab.set_title(1, 'Histogram')
        tab.set_title(2, 'Compare images')
        tab.set_title(3, 'Compare ZOOM')
        tab.set_title(4, 'PSNR')

        dashboard_layout = widgets.Layout(height='auto', overflow='auto', flex='1', max_height='100000px')
        dashboard = widgets.VBox([input_widgets, display_layout, tab], layout=dashboard_layout)

        display(dashboard)
        display(self.save_btn)
        display(self.txtbox)

    def dropdown_eventhandler(self, change):
        self.update_plots()
    def update_files_channel(self):
        channel = self.dropdown_ch_name.value
        self.files_channel = [file for file in self.result if str(channel + '.ome.tiff') in file]
        self.dropdown_image_names.options = ['None'] + [os.path.basename(f) for f in self.files_channel]

    def update_plots(self):
        self.plot_output_histogram.clear_output()
        self.plot_output_images.clear_output()
        self.plot_output_comparison.clear_output()
        self.plot_output_ZOOM.clear_output()
        self.plot_output_psnr.clear_output()

        self.update_files_channel()
        # files_channel = [file for file in self.result if str(self.dropdown_ch_name.value + '.ome.tiff') in str(file)]
        # self.files_channel = files_channel

        sample_images = self.dropdown_sample_images.value
        sample_images_number = self.dropdown_sample_images_number.value
        files_channel = self.files_channel

        if self.dropdown_image_names.value and self.dropdown_image_names.value != 'None':
            files_channel = [file for file in self.files_channel if self.dropdown_image_names.value in file]
        else:
            if sample_images == 'random':
                randomlist = random.sample(range(0, len(files_channel)), sample_images_number)
                files_channel = [files_channel[i] for i in randomlist]
            elif sample_images == 'top':
                files_channel = files_channel[:sample_images_number]
            elif sample_images == 'bottom':
                files_channel = files_channel[-sample_images_number:]

        self.files_displayed = files_channel


        imgs_channel, norm_imgs_channel, imgs_filtered = JN.calculus_fun(files_channel, self.dropdown_percentile.value,
                                                                         self.dropdown_th.value)

        with self.plot_output_histogram:
            self.set_params(res=50)
            JN.histogram_one_channel_all_img(norm_imgs_channel)
            JN.histogram_one_channel_all_img(imgs_filtered)

        with self.plot_output_images:
            self.plot_percentile(imgs_filtered, self.dropdown_res.value)
            print("Current filenames:")
            print(self.files_displayed)

        with self.plot_output_comparison:
            self.plot_comparison(imgs_channel, imgs_filtered, self.dropdown_res.value,
                                 self.dropdown_high_constrast_compare.value)

        with self.plot_output_ZOOM:
            self.plot_comparison_zoom(imgs_channel, imgs_filtered, self.dropdown_res.value,
                                      self.dropdown_high_constrast_compare.value)

    def plot_percentile(self, images, RES=300):
        self.set_params(res=RES)
        COL_PLOT = 1
        if COL_PLOT > 1:
            JN.plot_one_channel_side_by_side(images, columns=COL_PLOT, figsize=(70, 70), cmap=plt.cm.gray)
        else:
            JN.plot_one_channel(images, cmap=plt.cm.gray)



    def plot_comparison(self, images_or, imgs_filtered, res_compare, high_contrast):
        self.set_params(res=res_compare)
        JN.plot_compare_images(images_or, imgs_filtered, cmap=plt.cm.gray, high_contrast=high_contrast)

    def plot_comparison_zoom(self, images_or, imgs_filtered, res_compare, high_contrast):
        self.set_params(res=res_compare)
        JN.plot_compare_images_plotly(images_or, imgs_filtered, cmap=plt.cm.gray, high_contrast=high_contrast)

    def btn_eventhandler(self, obj):
        path_for_results = self.txtbox.value
        self.save_imgs(path_for_results)

    def btn_bin_eventhandler(self, obj):
        path_for_results = self.txtbox.value
        self.save_imgs(path_for_results, binary_masks=True)

    def save_imgs(self, path_for_results, binary_masks=False):
        if not os.path.exists(path_for_results):
            os.makedirs(path_for_results)

        files_channel = [file for file in self.result if str(self.dropdown_ch_name.value + '.ome.tiff') in str(file)]
        paths_save = [str(path_for_results + os.path.basename(os.path.dirname(sub))) for sub in files_channel]
        names_save = [str(path_for_results + os.path.basename(os.path.dirname(sub)) + '/' + os.path.basename(sub)) for
                      sub in files_channel]

        imgs_channel, norm_imgs_channel, imgs_filtered = JN.calculus_fun(files_channel,
                                                                         self.dropdown_percentile.value,
                                                                         self.dropdown_th.value,
                                                                         )
        if binary_masks:
            imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]

        images_test = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
        it = list(images_test)
        print('Images saved at ', path_for_results)
        print(f'Channel: {self.dropdown_ch_name.value}, '
              f'Percentile: {self.dropdown_percentile.value}, thresholding: {self.dropdown_th.value}')


