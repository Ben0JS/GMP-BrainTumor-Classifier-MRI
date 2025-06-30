-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jun 20, 2022 at 08:27 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `brain_tumor`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `recommend`
--

CREATE TABLE `recommend` (
  `id` int(11) NOT NULL,
  `btype` varchar(30) NOT NULL,
  `details` varchar(100) NOT NULL,
  `hospital` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `recommend`
--

INSERT INTO `recommend` (`id`, `btype`, `details`, `hospital`) VALUES
(1, 'Glioma Tumor', 'Blueberries, Turmeric', 'Neurosurgeon in Chennai Center for Brain & Spine, Chennai'),
(2, 'Meningioma Tumor', 'Broccoli, Pumpkin seeds', 'Kovai Medical Center , Coimbatore'),
(3, 'Pituitary Tumor', 'Dark chocolate, Nuts', 'Neuro Life Hospital ');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `gender` varchar(10) NOT NULL,
  `address` varchar(50) NOT NULL,
  `dob` varchar(15) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `rdate` varchar(15) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `gender`, `address`, `dob`, `mobile`, `email`, `uname`, `pass`, `rdate`) VALUES
(1, 'Raj', 'Male', '11,trichy', '11-08-1997', 9012388432, 'raja@myinfo.in', 'raj', 'Raj@1234', '10-02-2020'),
(2, 'Raj', 'Male', '33,KS Nagar', '11-09-1998', 9054621096, 'raj@gmail.com', 'raj', '1234', '19-06-2022');
