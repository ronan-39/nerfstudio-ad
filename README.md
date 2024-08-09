A fork of Nerfstudio to add feature rendering.

### Setup
Follow the regular installation instructions, and use this repo to install nerfstudio
in a new conda environment https://docs.nerf.studio/quickstart/installation.html
    ^This mainly consists of the following:
    ```
    cd nerfstudio
    pip install --upgrade pip setuptools
    pip install -e .
    ```

#### Generating Training Data:
Note that training data does not have to be in the repo's folder anymore, I'm trying to keep it outside to be more organized.
1. Train NeRFs with the following command:
    `ns-train nerfacto --pipeline.model.implementation torch --experiement-name [some_name] instant-ngp-data --data {path to dir *containing* transforms.json}`
    Replacing [some_name].
2. After the NeRF is trained, you have to modify the config.yml found at .../outputs/[some_name]/nerfacto/xxx-xx-xx_xxxxx/config.yml:
    On line 88, add the absolute path to the outputs folder. For MAD data, it will look something like:
        - /
        - home
        - username
        - Documents
        - MAD-Sim
        - XXAnimal
        - outputs
    On the new line 107, do the same thing, except exclude '- outputs':
        - / 
        - ...
        - XXAnimal
This step can be skipped if you train the NeRFs in the repo's folder, so its up to you if you want to do it like that. Just make sure to ignore the dataset folders.
3. To generate training data, follow the pattern of uncommented code in main.py. num_images is the number of images that will be generated. All images will be uniformly sampled from a sphere around the object. It's setup to work for MAD, since the origin of the NeRF in MAD is the origin of the object, but this may not be the case for real data.

If you test this its important to note that it seems as though Nerfacto fails to train for many models when using pytorch. The MAD animals I have confirmed to train are gorilla, turtle, zalika. Bird and owl train partially, but have an unusual amount of artifacts. The rest do not train at all--they result in a completely transparent field, for some reason.