if [ ! -d "./images" ]; then
  kaggle datasets download -d splcher/animefacedataset
  echo "Unzipping the files..."
  unzip animefacedataset.zip > /dev/null
  rm animefacedataset.zip
  echo "Done!"
fi
